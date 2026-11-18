"""Pearl-C4, an experimental PoUW variant (README.md).  Importing it registers its scheme, ``pearl-c-nvfp4-v0``, with
PoUW's registry (`verity.protocols.resource_accounting.compute.pouw.prover.schemes.register`), which names it and refuses it until then."""
from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.schemes import register


def _pearl_c4(fmt: str, *, hashing: str):
    def make(**config: Any):
        from .prover.pearl_c4 import PearlC4
        return PearlC4(fmt, hashing=hashing, **config)
    return make


register("pearl-c-nvfp4-v0", _pearl_c4("nvfp4", hashing="h0"))
