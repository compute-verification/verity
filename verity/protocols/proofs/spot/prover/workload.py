"""SPOT's messages between the workload and the prover: what SPOT's prover asks of the workload, and what the workload
hands SPOT's prover. Both parties are the developer's, so none of these messages reaches the auditor. With PoUW's,
PoUS's and the network attester's (each protocol's `prover/workload.py`), they are the one list
`tests/test_workload_messages.py` holds: each message declared once, named in the Glossary's words, with its writer and
its reader. A message more than one protocol reads is declared here. `tests/workload_messages.toml` lists today's
writers and readers by lane.

~~~text
prover   -> workload   RegisteredCompiler, RegisteredWeights, CommitmentScheme, CanonicalServing,
                       MonitoringPeriodDeclaration
workload -> prover     Circuit, ServingConfiguration, Run, Request, RunTotals, Advice, BoundaryCommitment,
                       RequestProgram, Values
~~~

The architecture's adapter hands the prover the compiler and the weights up front. Today the prover hands them to the
workload (`RegisteredCompiler`, `RegisteredWeights`), and they are declared in today's direction. The test names both
as the two exceptions.
"""

from __future__ import annotations

from verity.channel import Body, Schema
from verity.protocol import Channel, Role

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))
#: A body's bound: the wire's u32 length.
MAX_BODY = (1 << 32) - 1
BODY = Body(max=MAX_BODY)

MESSAGES = (
    Schema("RegisteredCompiler", 1, "structure", body=BODY),
    Schema("RegisteredWeights", 1, "commitment", body=BODY),
    Schema("CommitmentScheme", 1, "structure", body=BODY),
    Schema("CanonicalServing", 1, "structure", body=BODY),
    Schema("MonitoringPeriodDeclaration", 1, "structure", body=BODY),
    Schema("Circuit", 1, "structure", body=BODY),
    Schema("ServingConfiguration", 1, "structure", body=BODY),
    Schema("Run", 1, "commitment", body=BODY),
    Schema("Request", 1, "private", body=BODY),
    Schema("RunTotals", 1, "structure", body=BODY),
    Schema("Advice", 1, "private", body=BODY),
    Schema("BoundaryCommitment", 1, "commitment", body=BODY),
    Schema("RequestProgram", 1, "structure", body=BODY),
    Schema("Values", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(PROVER, WORKLOAD, ("RegisteredCompiler", "RegisteredWeights", "CommitmentScheme", "CanonicalServing",
                               "MonitoringPeriodDeclaration")),
    Channel(WORKLOAD, PROVER, ("Circuit", "ServingConfiguration", "Run", "Request", "RunTotals", "Advice",
                               "BoundaryCommitment", "RequestProgram", "Values")),
)
