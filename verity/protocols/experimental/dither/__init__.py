"""Dither, an experimental PoUW variant (PROTOCOL.md).  Importing it registers its scheme, ``dither-v3.2``, with PoUW's
registry (`verity.protocols.resource_accounting.compute.pouw.prover.schemes.register`)."""
from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.schemes import register


def _dither(**config: Any):
    from .prover.scheme import Dither
    return Dither(**config)


register("dither-v3.2", _dither)
