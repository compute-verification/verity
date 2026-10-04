"""Onsite Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps and streams
them over a fleet of two devices it keeps busy, the verifier's hardware at the site replays, counts, times and watches
the link, and `run` prints the verifier's view.

    uv run python examples/onsite.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.accounting.coverage.compute import Compute
from verity.accounting.coverage.memory import Memory
from verity.accounting.coverage.network import Network
from verity.accounting.integrity.replay import Replay
from verity.accounting.utilization.space import Space, encode
from verity.accounting.utilization.work import Work
from verity.compliance.nci import Nci
from verity.onsite import Onsite
from verity.primitives import partition as P
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, P.whole(circuit), advice_bits=2)

onsite = Onsite(
    integrity=Replay(k=3, delta=0.01),
    work=Work(difficulty=3, delta=0.01),
    space=Space(seed=b"weights", blocks=2048, k=16, deadline=0.001, rho=18 / 19, delta=0.01),
    compute=Compute(peak={"gpu": 1000.0}),
    memory=Memory(size={"gpu": 32 * 1024}),
    network=Network(max_timing_bits=7, side_bits=1000),
    nci=Nci(gamma=4096, alpha=0.2, advice=8, headroom=0.3),
)
registration, executions, inputs = executed(builder, schedule(requests, (3, 1, 4), builder), onsite.integrity.scheme)
fleet = {"gpu": 2}
principals = {
    "verifier": onsite.verifier(),
    "prover": onsite.prover(registration, executions, inputs, encode(b"weights", 2048)),
    "replayer": onsite.replayer(),
    "pinger": onsite.pinger(fleet, work_seconds=tries(registration, executions) / (2 * 1000.0), space_seconds=0.0004),
    "warden": onsite.warden(bytes_out=len(requests) * len(circuit.outputs) // 8, timing_bits=7),
}
for event in run(principals, view="verifier"):
    show(event)
