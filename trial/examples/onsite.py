"""Onsite Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps and streams
them over a site of two devices and an uplink, the verifier's hardware at the site replays, counts, times and observes
the link, and `run` prints the verifier's view.

    uv run python examples/onsite.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.accounting.capacity import Device
from verity.accounting.coverage import Coverage
from verity.accounting.explained.integrity.replay import Replay
from verity.accounting.explained.integrity.sampling import Weighted
from verity.accounting.explained.space import Space, encode
from verity.accounting.explained.work import Window, Work
from verity.accounting.observed import Network
from verity.compliance.nci import Nci
from verity.onsite import Onsite
from verity.primitives import partition as P
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, P.whole(circuit), advice_bits=2)

onsite = Onsite(
    integrity=Replay(law=Weighted(gates=0.05, outputs=0.05), delta=0.01),
    work=Work(difficulty=3, delta=0.01),
    space=Space(seed=b"weights", blocks=2048, k=64, deadline=0.001, delta=0.01),
    network=Network(max_timing_bits=7, side_bits=1000),
    coverage=Coverage(
        census={"gpu": Device(work=1000.0, memory=32 * 1024), "uplink": Device(egress=1000.0)},
        resources=("compute", "memory", "network"),
    ),
    nci=Nci(gamma=4096, alpha=0.2, max_headroom=0.3, builders=frozenset({builder.id()})),
)
stream, transcripts = executed(builder, schedule(requests, (3, 1, 4), builder), onsite.integrity.scheme)
site = {"gpu": 2, "uplink": 1}
principals = {
    "verifier": onsite.verifier(),
    "prover": onsite.prover(stream, transcripts, encode(b"weights", 2048)),
    "replayer": onsite.replayer(),
    "pinger": onsite.pinger(site, Window(0.0, tries(stream) / (2 * 1000.0)), space_seconds=0.0004),
    "warden": onsite.warden(bytes_out=len(requests) * len(circuit.outputs) // 8, timing_bits=7),
}
for event in run(principals, view="verifier"):
    show(event)
