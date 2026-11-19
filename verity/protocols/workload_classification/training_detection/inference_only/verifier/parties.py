"""The inference-only policy's manifests on the deployment layer (`verity.runtime`): each protocol's one party, the
verifier, which a deployment configuration's table declares from here. They are the manifests alone.

The policy has no rounds and reads no challenges: its check answers from the registered circuit, its division into
inference blocks and the registered information the verifier is handed. So nothing is sent, and a manifest has no
channel and no message. No party of it runs under the layer's runtime either. The check is Lean, the Lake package in
this directory. The verifier's machine runs it as one of the configuration's listed subprotocols' checks: its
``component`` capability, as SPOT's verifier names it.

The verifier (`VERIFIER`) is SPOT's party, the auditor's, one of it. This package imports no protocol, so its name,
principal and cardinality are stated here, and its test holds them to SPOT's declaration, and `ID` to the id SPOT
lists the policy under.

Its protocol is a core protocol of its executable (``EXE check PARAMS COMMITMENT EVIDENCE [BLOB...]``): `ID`,
``verity/inference-only/v0`` (`MANIFEST`, executable ``inference-only``, `InferenceOnlyCheck.Protocol`), the check on a
Program that holds every weight it reads and every wire between its units, counted in the Program's gates. Its params
are ``alpha``, ``gamma``, ``log``, ``parameters`` and ``program``. The composites list it.
"""

from __future__ import annotations

from verity.protocol import Manifest, Role

ID = "verity/inference-only/v0"
VERSION = 1
VERIFIER = "verifier"
_PARTIES = (Role(VERIFIER, "auditor", needs=("component",)),)
MANIFEST = Manifest(ID, VERSION, "inference-only check PARAMS", parties=_PARTIES, channels=(), messages=())

__all__ = ["ID", "MANIFEST", "VERIFIER", "VERSION"]
