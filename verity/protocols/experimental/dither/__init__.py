"""Dither, an experimental PoUW variant (PROTOCOL.md).  Importing it registers the builder of its family, ``dither``,
with PoUW's builders (`verity.protocols.resource_accounting.compute.pouw.prover.schemes.register`), so that its kernel
scheme builds by name at a card's parameters (`schemes.get("dither", card=..., pinned=...)`)."""
from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.schemes import register


def _dither(**config: Any):
    from .prover.pouw_scheme import Dither
    return Dither(**config)


register("dither", _dither)
