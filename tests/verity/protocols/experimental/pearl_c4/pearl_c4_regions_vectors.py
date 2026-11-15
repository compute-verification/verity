"""Pinned vectors for Pearl-C4's tile check's public input regions, `pouw_regions_vectors`' on Pearl-C4's anchors:
``uv run python tests/verity/protocols/experimental/pearl_c4/pearl_c4_regions_vectors.py`` rewrites
``tests/vectors/pearl_c4_regions.json``.  A change to that file is a change to a region's layout or derivation and needs a
reason in the PR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "resource_accounting" / "compute" / "pouw" / "verifier"))
import pouw_regions_vectors as RV
from verity.protocols.experimental.pearl_c4.prover import pc4
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.anchors import TileAnchors
from verity.protocols.experimental.pearl_c4.prover import pearl_c4 as F

OUT = Path(__file__).with_name("vectors") / "pearl_c4_regions.json"
#: (case, circuit, k, device), as `pouw_regions_vectors.CASES`
CASES = (("pc4-64", "pc4", 64, None),)
#: (case, circuit, k, device, row_k): the cut's shape (`pc4_rowk`), as `pouw_regions_vectors.ROWK_CASES`
ROWK_CASES = (("pc4-rowk-128", "pc4", 128, None, 64),)


def anchors(case: str, circuit: str, k: int, dev: str | None, row_k: int | None = None) -> TileAnchors:
    return TileAnchors.of(F.PearlC4("nvfp4", hashing="h0"), RV.context(case, k), pc4.Shape(2, k, 2, tile=(1, 1), row_k=row_k),
                          RV.ROWS, RV.COLS)


def vectors() -> dict:
    return RV.vectors(CASES, ROWK_CASES, anchors)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
