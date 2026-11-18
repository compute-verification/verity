"""Pearl-C4, an experimental PoUW variant (README.md).  Importing it registers the builder of its family, ``pearl-c4``,
with PoUW's builders (`verity.protocols.resource_accounting.compute.pouw.prover.schemes.register`): PoUW's kernel schemes
name ``pearl-c-nvfp4-v0`` and refuse to build it until then."""
from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.schemes import register


def _pearl_c4(**config: Any):
    from .prover.pearl_c4 import PearlC4
    return PearlC4(**config)


register("pearl-c4", _pearl_c4)
