"""Remote Verity end to end on a real circuit: the prover serves eight requests, and `run` prints the verifier's view.

    uv run python examples/remote.py [f32_add|bf16_gt_strict]
"""

import sys

from common import load, show

from verity.primitives import partition as P
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.integrity.proofs.transparent import Transparent
from verity.protocols.nci import Nci
from verity.protocols.pouw import Pouw
from verity.protocols.remote import Remote
from verity.run import run

circuit, requests, baseline = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")

remote = Remote(
    integrity=Proofs(k=64, max_width=16, delta=0.01, system=Transparent()),
    pouw=Pouw(difficulty=6, delta=0.01),
    nci=Nci(gamma=64, alpha=4.0, baseline=baseline),
)
actors = {
    **remote.verifier(),
    **remote.prover(circuit, P.greedy(circuit, 16), requests, capacity=2 * len(requests) * len(circuit.gates)),
}
for event in run(actors, view="verifier"):
    show(event)
