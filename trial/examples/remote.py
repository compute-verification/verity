"""Remote Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps, it streams
each step's advice and commitment over a fleet of two devices it keeps busy, and `run` prints the verifier's view.

    uv run python examples/remote.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.accounting.coverage.compute import Compute
from verity.accounting.integrity.proofs.construction import Proofs
from verity.accounting.integrity.proofs.transparent import Transparent
from verity.accounting.utilization.work import Work
from verity.compliance.nci import Nci
from verity.primitives import partition as P
from verity.remote import Remote
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, P.greedy(circuit, 16), advice_bits=2)

remote = Remote(
    integrity=Proofs(k=64, max_width=16, delta=0.01, system=Transparent()),
    work=Work(difficulty=3, delta=0.01),
    compute=Compute(peak={"gpu": 1000.0}),
    nci=Nci(gamma=64, alpha=4.0, advice=8, headroom=0.4, builders=frozenset({builder.id()})),
)
registration, executions, inputs = executed(builder, schedule(requests, (3, 1, 4), builder), remote.integrity.scheme)
fleet = {"gpu": 2}
principals = {
    "verifier": remote.verifier(seconds=tries(registration, executions) / (2 * 1000.0)),
    "prover": remote.prover(registration, executions, inputs, fleet),
}
for event in run(principals, view="verifier"):
    show(event)
