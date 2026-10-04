"""Integrity alone, end to end on a real circuit: the prover's scheduler batches eight requests into steps, it streams
each step's advice and commitment, and `run` plays the actors against each other and prints the verifier's view.

    uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]

Replay draws whole executions and has a sealed replayer run them again; proofs draw regions and have the prover prove
them, with the transparent proof system or SP1.
"""

import sys

from common import Batched, executed, load, schedule, show

from verity.primitives import circuits as C
from verity.primitives.randomness import Weighted
from verity.protocols.accounting.integrity import Integrity, serve
from verity.protocols.verification.proofs.protocol import Proofs
from verity.protocols.verification.proofs.transparent import Transparent
from verity.protocols.verification.replay import Replay
from verity.run import run

circuit_name, construction = (sys.argv[1:] + ["f32_add", "replay"][len(sys.argv) - 1 :])[:2]
circuit, requests = load(circuit_name)
builder = Batched(circuit, C.greedy(circuit, 16), advice_bits=2)

if construction == "replay":
    integrity = Integrity(Replay(), law=Weighted(gates=0.5, outputs=0.5), delta=0.01)
    others = {"replayer": [integrity.verification.replayer()]}
else:
    if construction == "sp1":
        from verity.protocols.verification.proofs.sp1.sp1 import Sp1 as system
    else:
        system = Transparent
    law = Weighted(gates=0.05, outputs=0.05)
    integrity, others = Integrity(Proofs(system()), law=law, delta=0.01, max_width=16), {}

stream, transcripts = executed(builder, schedule(requests, (3, 1, 4), builder), integrity.scheme)
principals = {
    "verifier": [integrity.verifier(), integrity.verification.verifier()],
    "prover": [serve(stream), integrity.prover(stream, transcripts)],
    **others,
}
for event in run(principals, view="verifier"):
    show(event)
