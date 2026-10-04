"""Remote Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps, it streams
each step's advice and commitment over a site of two devices it keeps busy, and `run` prints the verifier's view.

    uv run python examples/remote.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.accounting.capacity import Device
from verity.accounting.coverage import Coverage
from verity.accounting.explained.integrity.proofs.construction import Proofs
from verity.accounting.explained.integrity.proofs.transparent import Transparent
from verity.accounting.explained.integrity.sampling import Weighted
from verity.accounting.explained.work import Window, Work
from verity.compliance.nci import Nci
from verity.primitives import partition as P
from verity.remote import Remote
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, P.greedy(circuit, 16), advice_bits=2)

remote = Remote(
    integrity=Proofs(law=Weighted(gates=0.05, outputs=1.0), max_width=16, delta=0.01, system=Transparent()),
    work=Work(difficulty=3, delta=0.01),
    coverage=Coverage(census={"gpu": Device(work=1000.0)}, resources=("compute",)),
    nci=Nci(gamma=4096, alpha=0.2, max_headroom=0.4, builders=frozenset({builder.id()})),
)
stream, transcripts = executed(builder, schedule(requests, (3, 1, 4), builder), remote.integrity.scheme)
site = {"gpu": 2}
principals = {
    "verifier": remote.verifier(Window(0.0, tries(stream) / (2 * 1000.0))),
    "prover": remote.prover(stream, transcripts, site),
}
for event in run(principals, view="verifier"):
    show(event)
