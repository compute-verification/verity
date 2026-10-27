"""The remote composite (`verity/remote/v0`: sampled proofs and PoUW) on the lifecycle, as a deployment of two parties.

The developer serves its own load, so nothing the auditor runs sees the system's I/O: the developer reports what it served
through its declared advice, and the audit holds each declaration block to its committed values. Work it leaves out of
its declarations is not credited and not detected (`unreported` is accepted): the credit is a lower bound on the work done.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from lifecycle import (DEVELOPER, LATER, OUTSIDE, PLAN, SHARED, SIZES, VERIFIER, Developer, Device, Layout, Link, Store,
                       Verifier, load_of, products, theta_of)
from verity.core.deploy import runtime
from verity.core.deploy.manifest import Channel, Manifest, Role

ID = "lifecycle-remote"
LAYOUT = Layout("remote", ("node0",), (Device("gpu0", "sim-int8-a", "node0"), Device("gpu1", "sim-int8-b", "node0")),
                (Store("nvme0", "node0", 1 << 43),), (Link("uplink0", ("node0", OUTSIDE), 100_000, 100_000),))
"""The site's first hardware registration: one node with two devices and a store, and its uplink, with none of the
auditor's devices on it."""

MANIFEST = Manifest(
    ID, 1, "examples/lifecycle/remote.py:MANIFEST",
    parties=(Role(DEVELOPER, "developer", needs=("coins",)), Role(VERIFIER, "auditor", needs=("coins",))),
    channels=(Channel(DEVELOPER, VERIFIER, ("Hardware", "Register", "Declare", "Proofs")),
              Channel(VERIFIER, DEVELOPER, ("Draw", "Done"), carries_coins=True)),
    messages=SHARED)
PROTOCOL = runtime.ProtocolDef(MANIFEST, {DEVELOPER: Developer, VERIFIER: Verifier})


def run(cheat: str | None = None, seed: bytes = b"lifecycle/remote") -> dict[str, Any]:
    """One monitoring period of the remote composite, simulated: the verifier's outcome, and each party's."""
    config = {"protocol": ID, "params": {"protocol": "remote", "declarations": len(SIZES)},
              "party": [{"name": DEVELOPER}, {"name": VERIFIER}]}
    caps = {DEVELOPER: {"theta": theta_of(1), "state0": 0, "layout": LAYOUT, "later": replace(LATER, site="remote"),
                        "plan": PLAN, "product": products()["remote"][0], "load": load_of(2, SIZES), "cheat": cheat}}
    result = runtime.run(config, [DEVELOPER, VERIFIER], "simulated", None, protocols={ID: PROTOCOL}, seed=seed,
                         authenticate=False, capabilities=caps)
    return {"status": result.status, "outcomes": result.outcomes, **(result.outcomes[VERIFIER]["outcome"] or {})}
