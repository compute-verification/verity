"""Onsite Verity end to end on a real circuit: the prover's scheduler batches eight requests into steps and streams
them over a site of two devices and an uplink, the verifier's hardware at the site replays, counts, times and observes
the link, and `run` prints the verifier's view.

    uv run python examples/onsite.py [f32_add|bf16_gt_strict]
"""

import sys

from common import Batched, executed, load, schedule, show, tries

from verity.primitives import circuits as C
from verity.primitives.physical import Device, Window
from verity.primitives.randomness import Weighted
from verity.protocols.accounting.communication import Communication
from verity.protocols.accounting.integrity import Integrity
from verity.protocols.accounting.reconcile import Reconcile
from verity.protocols.accounting.space.pous import Pous, encode
from verity.protocols.accounting.work.pouw import Pouw
from verity.protocols.compliance.nci import Nci
from verity.protocols.onsite import Onsite
from verity.protocols.verification.replay import Replay
from verity.run import run

circuit, requests = load(sys.argv[1] if len(sys.argv) > 1 else "f32_add")
builder = Batched(circuit, C.whole(circuit), advice_bits=2)
census = {"gpu": Device(work=1000.0, memory=32 * 1024), "uplink": Device(egress=1000.0)}

onsite = Onsite(
    integrity=Integrity(Replay(), law=Weighted(gates=0.05, outputs=0.05), delta=0.01),
    work=Pouw(difficulty=3, delta=0.01),
    space=Pous(seed=b"weights", blocks=2048, k=64, delta=0.01),
    communication=Communication(max_timing_bits=7),
    reconcile=Reconcile(("work", "space", "communication"), census, deadline=0.001, side_bits=1000),
    nci=Nci(gamma=4096, alpha=0.2, max_unexplained=0.3, builders=frozenset({builder.id()})),
)
stream, transcripts = executed(builder, schedule(requests, (3, 1, 4), builder), onsite.integrity.scheme)
site = {"gpu": 2, "uplink": 1}
window = Window(0.0, tries(stream) / (2 * 1000.0))
principals = {
    "verifier": onsite.verifier(),
    "prover": onsite.prover(stream, transcripts, encode(b"weights", 2048)),
    "replayer": onsite.replayer(),
    "pinger": onsite.pinger(site, window, space_window=Window(0.01, 0.0104)),
    "warden": onsite.warden(bytes_out=len(requests) * len(circuit.outputs) // 8, timing_bits=7, window=window),
}
for event in run(principals, view="verifier"):
    show(event)
