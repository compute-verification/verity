"""PoUW's H1 + fx1 tile against the circuit model's census: `Security/Proofs/Accounting/Gates/Rows.lean`'s `h1Rows`.

The Lean is the definition and this file mirrors it on the tile's Program (`h1tile.PouwH1Tile` with `EXACT` the s8
accumulator over `BlackwellS8ImmaDot32_v2`, on the full grid). A row's dot steps and their MACs (`census.dotMacs`) are the
instances of the tile's s8 dot classes in census/v1, `k` each by property/v1's family table. It has no served and no tail
MACs: an H1 tile's served role is on its receipt's words, which take no MACs. Its committed and served words are the
receipt's `2·(KS + m)` and the `2·KS` after its segment checkpoints.

census/v1's own MAC counters agree with the Lean on every row:

* its classes' products (`PRODUCTS`) are the row's MACs, `z`'s `r·c·K` and the lanes' segment dots' `r·c·(K + 32 m)`
  (`PouwSegDot_v2` and `PouwSegDot0_v2`, 32 each);
* it serves none of them and counts no tail: `GemmAccumulatorS8` and the segment dots count by `census.serves_none`, as
  the Lean's served MACs are those of the served role (`roleOf`), and no gate of the tile with MACs has that role.

Every class of the tile is listed (Rows.lean's `classed_h1Rows`, `h1Classes`) but its bodies, and property/v1 reads it, as
it does v3's, v4's, v5's and v6's.
"""
from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import PRODUCTS, census
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program, input_id
from verity.ml.protocols.pouw.circuit import h1tile as H

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment) are not on this tree")
ROWS = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "Gates" / "Rows.lean"
#: the lanes' Boolean segment dots, s8-family steps in property/v1's table (`PouwSegDot0` is `PouwSegDot` from 0)
LANE = ("PouwSegDot_v2", "PouwSegDot0_v2")
#: the tile's versions on this tree: v1, the one-stream v3, the gated v4 and J10-R's v5 and v6 (v1's and v3's
#: under call-scope noise)
TILES = {v: d for v, d in ((1, H.PouwH1Tile), *((v, getattr(H, f"PouwH1Tile{v}", None)) for v in (3, 4, 5, 6)))
         if d is not None}
#: a tile's classes that are bodies, not primitives: v1's and v5's exact accumulator (`EXACT`, whose type v3, v4 and v6
#: take and never call); the tile is its Program's root, not a class of its census
BODIES = {v: {"GemmAccumulatorS8_v2"} if v in (1, 5) else set() for v in TILES}
#: Lean's budgeted classes in property/v1's names
BUDGETED = {"rounding": "roundings", "compareSelect": "compare_select", "fold": "folds"}


def lean_block(name: str) -> str:
    """The list of `def name` in Rows.lean, as text."""
    block = ROWS.read_text().split(f"\ndef {name} ", 1)[1]
    return block[block.index(":= [") + 3:block.index("]\n\n") + 1]


def lean_rows(name: str) -> list:
    """The rows of `def name` in Rows.lean as Python values."""
    body = lean_block(name)
    for lean, py in ((r"\bsome ", ""), (r"\bnone\b", "None"), (r"\btrue\b", "True"), (r"\bfalse\b", "False")):
        body = re.sub(lean, py, body)
    rows = ast.literal_eval(body)
    assert rows, f"no rows read from {name}"
    return rows


def dot_k(family: str) -> int:
    """A class's products a step when it is an s8-family step, else 0."""
    step = P.FAMILY.get(family)
    return step.k if step else 0


@functools.cache
def tile_program(r: int, c: int, k: int, v: int = 1) -> Program:
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return Program(bind(TILES[v], EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c))))


@functools.cache
def tile_census(r: int, c: int, k: int, v: int = 1) -> dict:
    return census(tile_program(r, c, k, v))


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
    assert sum(cl["instances"] * PRODUCTS.get(cl["family"], 0) for cl in doc["classes"].values()) == macs
    t = doc["totals"]
    assert t["served_macs"] == t["tail_atoms"] == 0


def lean_class(fid: str, cls: str, sub: str) -> bool:
    """Whether property/v1's table has `fid` in Lean's class `cls` (`sub` a budgeted class's), an input by its id."""
    if cls == "input":
        return input_id(fid) is not None
    return P.class_of(fid) == (BUDGETED[sub] if cls == "budgeted" else cls)


@pytest.mark.slow
@pytest.mark.parametrize("v", sorted(TILES))
def test_h1_tile_is_classed(v):
    """`classed_h1Rows` on the tile's Program, each version on this tree (`TILES`), at 4 × 16, two lanes, so the XOR
    across lanes (`PouwXor32_v2`) is called: every class of its census is listed in property/v1's table (`class_of`, an
    input by its id) but its `BODIES`, and property/v1 refuses no gate of it as unlisted; v1's are Lean's `h1Classes` in
    theirs (the seeds' `PouwUnitSeeds_v2` and `PouwNoiseBlock_v2`, which `h1Dag` reads as the salt, too). Its dots are on
    bits, so the pass reads them down to their 1-bit gates and proves no s8-family window of the tile (`family` 0). Slow
    on a process's first tile (see `test_h1_census`); v3's, v4's, v5's and v6's catch a primitive of their one stream,
    dead-tile mask or call seeds and block counter that the registry lacks, which v1 never calls."""
    doc = tile_census(4, 16, 64, v)
    fams = {cl["family"] for cl in doc["classes"].values()}
    assert {f for f in fams if P.class_of(f) is None and input_id(f) is None} == BODIES[v]
    if v == 1:
        lean = re.findall(r'\("([^"]+)", \.(\w+)(?: \.(\w+))?', lean_block("h1Classes"))
        assert len(lean) >= 15 and all(lean_class(*e) for e in lean), [e for e in lean if not lean_class(*e)]
        assert {f for f, cls, _ in lean if cls not in ("input", "const")} | {"PouwUnitSeeds_v2", "PouwNoiseBlock_v2"} <= fams
    a = P.analyze(tile_program(4, 16, 64, v), doc)
    assert "unlisted" not in a.refused, a.refused["unlisted"]
    assert (a.family, a.macs) == (0, 0)


def test_the_word_segment_dots_prove_their_window():
    """property/v1 proves the word segment dots' window from s8 operands: `PouwSegDot0_v1` from 0, `PouwSegDot_v1` from an
    accumulator in window, each a family step of 32 MACs; `PouwSegDot_v1` on an accumulator of any word isn't proven."""
    s8 = (-128, 127)
    t0 = P.count(H.SG.PouwSegDot0Word, (True, True), [s8, s8])
    t = P.count(H.SG.PouwSegDotWord, (True, True, True), [(-(1 << 20), 1 << 20), s8, s8])
    assert (t0.family, t0.macs, t0.unproven) == (t.family, t.macs, t.unproven) == (1, 32, {})
    assert P.count(H.SG.PouwSegDotWord, (True, True, True)).unproven == {"PouwSegDot_v1": 1}
