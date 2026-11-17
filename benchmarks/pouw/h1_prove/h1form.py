"""This harness's H1 tile and statement form, ``H1_TILE`` and ``H1_TMPL`` (`settings.py`), handed to H1's lowering
(`verity.protocols.resource_accounting.compute.pouw.prover.lowering.h1unit`) as its arguments."""
from __future__ import annotations

from verity_config.settings import of

from verity.protocols.resource_accounting.compute.pouw.prover.lowering import h1unit as U

SETTINGS = of(__file__)


def tile() -> str:
    """``H1_TILE``: the H1 tile under the unit, v1 or v3 (`h1unit.tile`)."""
    return U.tile(SETTINGS.H1_TILE)


def layout() -> str | None:
    """``H1_TMPL``'s layout mode (`h1unit.layout`): None, the flat statement, unless it names a templated form."""
    return U.layout(SETTINGS.H1_TMPL)


def shape(k: int, r: int = U.TILE[0], c: int = U.TILE[1], j10r: bool = False) -> U.Shape:
    """`h1unit.Shape` at K, R × C (J10-R's with ``j10r``), over ``H1_TILE``'s tile."""
    return U.Shape(k, r, c, j10r, tile())


__all__ = ["layout", "shape", "tile"]
