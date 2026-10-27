"""The remote composite (`verity/remote/v0`: sampled proofs and PoUW) on the lifecycle, as a deployment of two parties.

The developer serves its own load, so nothing the auditor runs sees the system's I/O: the developer reports what it served
through its advice, and the audit holds each segment's circuit to its committed values. Work it leaves out of the advice is
not credited and not detected (`unreported` is accepted): the credit is a lower bound on the work done.
"""

from __future__ import annotations

from typing import Any

from lifecycle import (DEVELOPER, SHARED, VERIFIER, Developer, Device, Hardware, Verifier, load_of, products, theta_of)
from verity.core.deploy import runtime
from verity.core.deploy.manifest import Channel, Manifest, Role

ID = "lifecycle-remote"
SIZES = (3, 4, 2)
"""Requests in each segment of the example's period."""
HARDWARE = Hardware((Device("gpu0", "sim-int8-a"), Device("gpu1", "sim-int8-b")))

MANIFEST = Manifest(
    ID, 1, "examples/lifecycle/remote.py:MANIFEST",
    parties=(Role(DEVELOPER, "developer", needs=("coins",)), Role(VERIFIER, "auditor", needs=("coins",))),
    channels=(Channel(DEVELOPER, VERIFIER, ("Register", "Close", "Proofs")),
              Channel(VERIFIER, DEVELOPER, ("Draw", "Done"), carries_coins=True)),
    messages=SHARED)
PROTOCOL = runtime.ProtocolDef(MANIFEST, {DEVELOPER: Developer, VERIFIER: Verifier})


def run(cheat: str | None = None, seed: bytes = b"lifecycle/remote") -> dict[str, Any]:
    """One monitoring period of the remote composite, simulated: the verifier's outcome, and each party's."""
    config = {"protocol": ID, "params": {"protocol": "remote", "segments": len(SIZES)},
              "party": [{"name": DEVELOPER}, {"name": VERIFIER}]}
    caps = {DEVELOPER: {"theta": theta_of(1), "state0": 0, "hardware": HARDWARE, "product": products()["remote"][0],
                        "load": load_of(2, SIZES), "cheat": cheat}}
    result = runtime.run(config, [DEVELOPER, VERIFIER], "simulated", None, protocols={ID: PROTOCOL}, seed=seed,
                         authenticate=False, capabilities=caps)
    return {"status": result.status, "outcomes": result.outcomes, **(result.outcomes[VERIFIER]["outcome"] or {})}
