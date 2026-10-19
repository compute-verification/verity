"""PoUW's H1 + fx1 tile against the circuit model's census: `Security/Proofs/Accounting/Gates/Rows.lean`'s `h1Rows`.

The Lean is the definition and this file mirrors it on the tile's Program (`h1tile.PouwH1Tile` with `EXACT` the s8
accumulator over `BlackwellS8ImmaDot32_v2`, on the full grid). A row's dot steps and their MACs (`census.dotMacs`) are the
instances of the tile's s8 dot classes in census/v1, `k` each by property/v1's family table. It has no served and no tail
MACs: an H1 tile's served role is on its receipt's words, which take no MACs. Its committed and served words are the
receipt's `2·(KS + m)` and the `2·KS` after its segment checkpoints.

census/v1's own MAC counters disagree with the Lean on every row; the Lean is right, and the disagreements are pinned
here (census/v1 is circuits'):

* `served_macs + tail_atoms` misses the lanes' segment dots, `r·c·(K + 32 m)` MACs. `PRODUCTS` lists neither
  `PouwSegDot_v2` nor `PouwSegDot0_v2`, so census/v1 counts their gates as no products, though property/v1's family table
  classes `PouwSegDot_v1` as an s8 dot of 32. It counts only `z`'s `r·c·K`.
* `served_macs` counts `z`'s products as served: census/v1 counts every product as served unless a family rule
  (`FAMILY_COUNTS`, `S8Dot_v1`'s) splits it, while the Lean's served MACs are those of the served role (`roleOf`), and no
  gate of the tile with MACs has that role.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.core.protocols.pouw.circuit import h1tile as H

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
ROWS = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "Gates" / "Rows.lean"
#: the lanes' Boolean segment dots by the word dot property/v1's family table classes: `PouwSegDot0` is `PouwSegDot` from 0
LANE = {"PouwSegDot_v2": "PouwSegDot_v1", "PouwSegDot0_v2": "PouwSegDot_v1"}


def lean_rows(name: str) -> list:
    """The rows of `def name` in Rows.lean as Python values."""
    block = ROWS.read_text().split(f"\ndef {name} ", 1)[1]
    body = block[block.index(":= [") + 3:block.index("]\n\n") + 1]
    for lean, py in ((r"\bsome ", ""), (r"\bnone\b", "None"), (r"\btrue\b", "True"), (r"\bfalse\b", "False")):
        body = re.sub(lean, py, body)
    rows = ast.literal_eval(body)
    assert rows, f"no rows read from {name}"
    return rows


def dot_k(family: str) -> int:
    """A class's products a step when it is an s8-family step, else 0."""
    step = P.FAMILY.get(LANE.get(family, family))
    return step.k if step else 0


def tile_census(r: int, c: int, k: int) -> dict:
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return census(Program(bind(H.PouwH1Tile, EXACT=exact, R=r, C=c,
                               PAIRS=tuple((i, j) for i in range(r) for j in range(c)))))


@pytest.mark.slow
@pytest.mark.parametrize("r,c,k,out", lean_rows("h1Rows"))
def test_h1_census(r, c, k, out):
    """Slow on a process's first tile: building it traces its seeds' SHA-512 on bits (about 35 s), then about 1 s a row."""
    lay = H.Layout(r, c, k)
    doc = tile_census(r, c, k)
    dots = [(cl["family"], cl["instances"]) for cl in doc["classes"].values() if dot_k(cl["family"])]
    assert {f for f, _ in dots} == {"PouwSegDot_v2", "PouwSegDot0_v2", "BlackwellS8ImmaDot32_v2"}
    macs = sum(n * dot_k(f) for f, n in dots)
    assert out == (sum(n for _, n in dots), macs, 0, 0, 2 * lay.sets, 2 * lay.ks)
    lane = sum(n * dot_k(f) for f, n in dots if f in LANE)
    assert lane == r * c * (k + 32 * lay.m) and macs - lane == r * c * k
    t = doc["totals"]
    assert t["served_macs"] + t["tail_atoms"] == macs - lane
    assert t["served_macs"] == r * c * k
