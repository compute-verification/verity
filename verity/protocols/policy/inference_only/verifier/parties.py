"""The inference-only policy's manifest on the deployment layer (`verity.runtime`): its one party, the
verifier, which a deployment configuration's table declares from here. It is the manifest alone.

The policy has no rounds and reads no challenges: its check answers from the registered circuit, its division into
inference blocks and the registered information the verifier is handed. So nothing is sent, and the manifest has no
channel and no message. No party of it runs under the layer's runtime either. The check is Lean, the Lake package in
this directory, whose executable is ``inference-only``. The verifier's machine runs it as one of the configuration's
listed subprotocols' checks: its ``component`` capability, as SPOT's verifier names it.

The verifier (`VERIFIER`) is SPOT's party, the auditor's, one of it. This package imports no protocol, so its name,
principal and cardinality are stated here, and its test holds them to SPOT's declaration, and `ID` to the id SPOT
lists the policy under. The parameters are the core protocol's: the ``PARAMS`` its Lean executable's check reads, the
core command line ``inference-only check PARAMS COMMITMENT EVIDENCE`` (``alpha``, ``gamma``, ``log``, ``parameters``,
``program``).
"""

from __future__ import annotations

from verity.protocol import Manifest, Role

ID = "verity/inference-only/v0"
VERSION = 1
VERIFIER = "verifier"
MANIFEST = Manifest(ID, VERSION, "inference-only check PARAMS",
                    parties=(Role(VERIFIER, "auditor", needs=("component",)),), channels=(), messages=())

__all__ = ["ID", "MANIFEST", "VERIFIER", "VERSION"]
