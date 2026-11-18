"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). A tile that passes
the tile rule costs its credited cells at the record's kept width |K| (`Credit.lean`'s `creditAtAssumption`), so a
block's cost reads the record's kept width (``K_KEPT``, `UnitRecord.k_kept`) beside the tile's TM × TN, not the
Definition's K, which is the unit's padded width. The cost is of a tile whose every cell is credited: filler, excluded and
given-up rows credit none (`UnitRecord.credited`).
"""

from __future__ import annotations

from collections.abc import Mapping

from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Scheme, register


def tile_macs(shape: Mapping[str, int]) -> int:
    return shape["TM"] * shape["TN"] * shape["K_KEPT"]


PEARL_PLUS = register(Scheme(
    name="pearl-plus",
    protocol="verity/pouw/pearl-plus/v1",
    resource="compute",
    blocks=(Block("PearlPlusTile_v1", tile_macs), Block("PearlCTileRead_v1", tile_macs)),
    assumptions=("Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1",),
    log=(Order("weightsRoot", "pouw/salt"), Order("commitment", "pouw/draw")),
    charges=(Charge("inert-work", "rho"), Charge("sample", "epsilon"), Charge("over-bound-tiles", "over_cap_draws"),
             Charge("excluded-rows", "give_up")),
    option="pouw:pearl-c-h100-v1-h3",
    hooks=("blocks", "declared"),
    requires={"one-labelling": "its own pass manifests, tied to the window record's pearl-a/, pearl-y/ and pearl-b/ values",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
))
