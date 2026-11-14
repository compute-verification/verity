"""Sampled proofs' messages with the workload: what the workload and the prover send each other, as message types
(`verity.channel.Schema`) and the two channels that carry them (`verity.protocol.Channel`), one a direction.

The workload (`WORKLOAD`) is the developer's computation behind its adapter, and the prover (`PROVER`) is the
developer's side of the protocols. In SPOT[Flock] on its own they are two parties, the inference server and the proof
server, and `parties.SPOT_MANIFEST` carries `REPLAY` between them. In the core deployment they are one party on one
machine, so its manifest has no channel between them, and the rest of these messages travel as files, records in a log
and function calls. `tests/test_workload_messages.py` holds the list: every protocol's declaration (a module of this
name in its `verifier/`), where the tree writes and reads each message today, and what crosses between the two and is
no message of the list.

A message that serves more than one protocol is declared here, once, since the proof system owns the commitments. A
protocol on top declares only its own messages and names these in its channels.

~~~text
workload -> prover
Circuit      the circuit the workload runs, cut into proof blocks, before any run
Settings     the workload's own statement of how it runs
Window       a closed execution phase: its commitment's root and what each protocol's hook made of it
Io           the run's external inputs and outputs
Summary      that the run ended with every record written
Advice       the execution phase's advice
Programs     the compiler's output on the advice: the Programs the execution phase ran
Values       every value the execution phase computed, committed
Replay       the values a sampled block's proof blocks read, with the salts that open their commitments

prover -> workload
Compiler     the registered compiler, by name and digest
Weights      the weights' registration
~~~

Each direction is the one the code has. A body is the bytes the writer writes today, and no message but `Replay` has a
bound of its own, so each takes the frame's (`MAX_BODY`).
"""

from __future__ import annotations

from verity.channel import Body, Int, Schema
from verity.protocol import Channel

WORKLOAD, PROVER = "workload", "prover"
MAX_BLOCKS = 1 << 16
MAX_BODY = MAX_REPLAY = (1 << 32) - 1
BLOCK = {"block": Int(0, MAX_BLOCKS - 1)}
"""The head of a message sent once an execution phase: which one."""
REPLAY = Schema("Replay", 1, "private", BLOCK, Body(max=MAX_REPLAY))
MESSAGES = (
    Schema("Circuit", 1, "structure", body=Body(max=MAX_BODY)),
    Schema("Settings", 1, "structure", body=Body(max=MAX_BODY)),
    Schema("Window", 1, "commitment", BLOCK, Body(max=MAX_BODY)),
    Schema("Io", 1, "private", body=Body(max=MAX_BODY)),
    Schema("Summary", 1, "structure", body=Body(max=MAX_BODY)),
    Schema("Advice", 1, "private", BLOCK, Body(max=MAX_BODY)),
    Schema("Programs", 1, "private", BLOCK, Body(max=MAX_BODY)),
    Schema("Values", 1, "private", BLOCK, Body(max=MAX_BODY)),
    REPLAY,
    Schema("Compiler", 1, "structure", body=Body(max=MAX_BODY)),
    Schema("Weights", 1, "commitment", body=Body(max=MAX_BODY)),
)
CHANNELS = (
    Channel(WORKLOAD, PROVER, ("Circuit", "Settings", "Window", "Io", "Summary", "Advice", "Programs", "Values", "Replay")),
    Channel(PROVER, WORKLOAD, ("Compiler", "Weights")),
)

__all__ = ["BLOCK", "CHANNELS", "MAX_BLOCKS", "MAX_BODY", "MAX_REPLAY", "MESSAGES", "PROVER", "REPLAY", "WORKLOAD"]
