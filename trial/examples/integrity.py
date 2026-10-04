"""Integrity alone, end to end on a real circuit: the prover's scheduler batches eight requests into steps, it streams
each step's advice and commitment, and `run` plays the actors against each other and prints the verifier's view.

    uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]

Replay samples whole executions and has a sealed replayer run them again; proofs sample regions and have the prover
prove them, with the transparent proof system or SP1.
"""

import sys

from common import Batched, executed, load, schedule, show

from verity.accounting.integrity.proofs.construction import Proofs
from verity.accounting.integrity.proofs.transparent import Transparent
from verity.accounting.integrity.replay import Replay
from verity.primitives import partition as P
from verity.primitives.registration import serve
from verity.run import run

circuit_name, construction = (sys.argv[1:] + ["f32_add", "replay"][len(sys.argv) - 1 :])[:2]
circuit, requests = load(circuit_name)
builder = Batched(circuit, P.greedy(circuit, 16), advice_bits=2)

if construction == "replay":
    integrity = Replay(k=2, delta=0.01)
    others = {"replayer": [integrity.replayer()]}
else:
    if construction == "sp1":
        from verity.accounting.integrity.proofs.sp1.sp1 import Sp1 as system
    else:
        system = Transparent
    integrity, others = Proofs(k=64, max_width=16, delta=0.01, system=system()), {}

registration, executions, inputs = executed(builder, schedule(requests, (3, 1, 4), builder), integrity.scheme)
principals = {
    "verifier": [integrity.verifier()],
    "prover": [serve(registration, executions), integrity.prover(registration, executions, inputs)],
    **others,
}
for event in run(principals, view="verifier"):
    show(event)
