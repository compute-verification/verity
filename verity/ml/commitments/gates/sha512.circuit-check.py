"""circuit-check's bindings of `verity.ml.commitments.gates.sha512`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _sha512_roots() -> list[Any]:
    """The one compression a Boolean circuit calls (the LMS verifier's one-time public key, the certifier's head check, the
    opening unit's row digests, V*'s openings and links, PoUS's row digests), each caller laying out its own bits into its
    words at the call, so its own correspondence and pin hold for every caller's compressions."""
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.sha512 import Sha512Compress

    return [bind(Sha512Compress)]
