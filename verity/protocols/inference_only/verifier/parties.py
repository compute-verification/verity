"""The inference-only policy's manifest on the deployment layer (`verity.configuration.deploy`): its one party, the
verifier, which a deployment configuration's table declares from here. It is the manifest alone.

The policy has no rounds and reads no coins (`NciCheck.Protocol`): its check answers from the declared circuit, the
declared division and the registered information the verifier is handed. So nothing is sent, and the manifest has no
channel and no message. No party of it runs under the layer's runtime either. The check is the Lean's (`nci/lean`, the
executable ``inference-only``), and the verifier's machine runs it as one of the configuration's listed subprotocols'
checks: its ``component`` capability, as sampled proofs' verifier names it.

The verifier (`VERIFIER`) is sampled proofs' party, the auditor's, one of it. This package imports no protocol, so its
name, principal and cardinality are stated here, and its test holds them to sampled proofs' declaration. The id and the
parameters are the Lean's: `ID` is `NciCheck.Protocol.protocolId`, and the manifest names the function that reads the
parameters (``alpha``, ``gamma``, ``log``, ``parameters``, ``program``).
"""

from __future__ import annotations

from verity.configuration.deploy.manifest import Manifest, Role

ID = "verity/inference-only/v0"
VERSION = 1
VERIFIER = "verifier"
MANIFEST = Manifest(ID, VERSION, "NciCheck.Protocol:paramsOf",
                    parties=(Role(VERIFIER, "auditor", needs=("component",)),), channels=(), messages=())

__all__ = ["ID", "MANIFEST", "VERIFIER", "VERSION"]
