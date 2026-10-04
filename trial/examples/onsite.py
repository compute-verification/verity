"""Onsite Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps and streams
them, the verifier's hardware at the site replays, measures and watches the link, and `run` prints the verifier's view.

    uv run python examples/onsite.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, load, schedule, show

from verity.primitives import partition as P
from verity.protocols.integrity.replay import Replay
from verity.protocols.nci import Nci
from verity.protocols.network import Network
from verity.protocols.onsite import Onsite
from verity.protocols.pous import Pous, encode
from verity.protocols.pouw import Pouw
from verity.protocols.transparency import Transparency
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, P.whole(circuit), advice_bits=2)

onsite = Onsite(
    integrity=Replay(k=2, delta=0.01),
    pouw=Pouw(difficulty=6, delta=0.01),
    nci=Nci(gamma=4096, alpha=0.2),
    pous=Pous(seed=b"weights", blocks=1024, k=16, deadline=0.001, rho=18 / 19, delta=0.01),
    transparency=Transparency(max_timing_bits=7),
    network=Network(side_bits=1000),
)
principals = {
    "verifier": onsite.verifier(),
    "prover": onsite.prover(builder, schedule(requests, (3, 1, 4), builder), encode(b"weights", 1024)),
    "replayer": onsite.replayer(),
    "pinger": onsite.pinger(compute=2 * len(requests) * len(circuit.gates), memory=64 * 1024, elapsed=0.0004),
    "warden": onsite.warden(bytes_out=8 * len(requests), timing_bits=7),
}
for event in run(principals, view="verifier"):
    show(event)
