"""Integrity alone, end to end on a real circuit: the prover serves eight requests, and `run` plays the actors against
each other and prints the verifier's view.

    uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]

Replay samples whole executions and has a sealed replayer run them again; proofs sample regions and have the prover
prove them, with the transparent proof system or SP1.
"""

import sys

from common import load, show

from verity.primitives import partition as P
from verity.primitives.events import Send, tell
from verity.primitives.registration import register
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.integrity.proofs.transparent import Transparent
from verity.protocols.integrity.replay import Replay
from verity.run import run

circuit_name, construction = (sys.argv[1:] + ["f32_add", "replay"][len(sys.argv) - 1 :])[:2]
circuit, requests = load(circuit_name)

if construction == "replay":
    integrity, partition = Replay(k=2, delta=0.01), P.whole(circuit)
    others = {"replayer": [integrity.replayer()]}
else:
    if construction == "sp1":
        from verity.protocols.integrity.proofs.sp1.sp1 import Sp1 as system
    else:
        system = Transparent
    integrity, partition, others = Proofs(k=64, max_width=16, delta=0.01, system=system()), P.greedy(circuit, 16), {}

registration = register(circuit, partition, requests, integrity.scheme)
principals = {
    "verifier": [integrity.verifier()],
    "prover": [tell(Send(registration, "verifier")), integrity.prover(requests, registration)],
    **others,
}
for event in run(principals, view="verifier"):
    show(event)
