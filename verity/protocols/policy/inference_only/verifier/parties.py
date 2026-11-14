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

Two protocols, each a core protocol of its executable (``EXE check PARAMS COMMITMENT EVIDENCE [BLOB...]``):

* `ID`, ``verity/inference-only/v0`` (`MANIFEST`, executable ``inference-only``, `NciCheck.Protocol`): the check on
  a Program that holds every weight it reads and every wire between its units, counted in the Program's gates. Its
  params are ``alpha``, ``gamma``, ``log``, ``parameters`` and ``program``. The composites list it.
* `ROWS_ID`, ``verity/inference-only/rows/v1`` (`ROWS_MANIFEST`, executable ``inference-only-rows``,
  `InferenceOnly.RowsProtocol`, a `Core.BlobProtocol`): the check on an execution phase's circuit record, each block
  counted in rows of its templates' circuit files. No composite lists it.
  - params: canonical JSON with exactly the keys ``gamma`` (Γ, the most rows a block holds), ``alpha`` (α as
    ``[num, den]``, ``den`` not 0, the most fresh bits a row), ``circuits`` (``[[template id, SHA-512], ...]``, one
    pair for each of the record's templates in its order, the digest its circuit file's in lower-case hex), ``log``
    (the registration records before the run's own, oldest first), ``pins`` (the closure digests of the Definitions
    the policy takes as readers of a table opening) and ``program`` (a ``verity-ir/descriptor/v1`` text);
  - commitment: the run's own registration record, read by v0's decoder: canonical JSON, an object;
  - evidence: canonical JSON ``{"record"}`` or ``{"order", "record"}``: the execution phase's circuit record, as the
    text the registration record commits, and the order of its unit ranges where the record states none;
  - blobs: each template's circuit file, and the pinned lookup tables a file looks up, each read only under a digest
    ``circuits`` names or a pinned table's own.
"""

from __future__ import annotations

from verity.protocol import Manifest, Role

ID = "verity/inference-only/v0"
ROWS_ID = "verity/inference-only/rows/v1"
VERSION = 1
VERIFIER = "verifier"
_PARTIES = (Role(VERIFIER, "auditor", needs=("component",)),)
MANIFEST = Manifest(ID, VERSION, "inference-only check PARAMS", parties=_PARTIES, channels=(), messages=())
ROWS_MANIFEST = Manifest(ROWS_ID, VERSION, "inference-only-rows check PARAMS", parties=_PARTIES, channels=(),
                         messages=())

__all__ = ["ID", "MANIFEST", "ROWS_ID", "ROWS_MANIFEST", "VERIFIER", "VERSION"]
