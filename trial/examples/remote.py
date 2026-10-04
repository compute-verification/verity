"""Remote Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps, it streams
each step's advice and commitment over a site of two devices it keeps busy, and `run` prints the verifier's view.

    uv run python examples/remote.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.primitives import circuits as C
from verity.primitives.physical import Device, Window
from verity.primitives.randomness import Weighted
from verity.protocols.accounting.integrity import Integrity
from verity.protocols.accounting.reconcile import Reconcile
from verity.protocols.accounting.work.pouw import Pouw
from verity.protocols.compliance.nci import Nci
from verity.protocols.remote import Remote
from verity.protocols.verification.proofs.protocol import Proofs
from verity.protocols.verification.proofs.transparent import Transparent
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, C.greedy(circuit, 16), advice_bits=2)
census = {"gpu": Device(work=1000.0)}

remote = Remote(
    integrity=Integrity(Proofs(Transparent()), law=Weighted(gates=0.05, outputs=1.0), delta=0.01, max_width=16),
    work=Pouw(difficulty=3, delta=0.01),
    reconcile=Reconcile(resources=("work",), census=census),
    nci=Nci(gamma=4096, alpha=0.2, max_unexplained=0.4, builders=frozenset({builder.id()})),
)
stream, transcripts = executed(builder, schedule(requests, (3, 1, 4), builder), remote.integrity.scheme)
site = {"gpu": 2}
principals = {
    "verifier": remote.verifier(Window(0.0, tries(stream) / (2 * 1000.0))),
    "prover": remote.prover(stream, transcripts, site),
}
for event in run(principals, view="verifier"):
    show(event)
