"""proved_ticks at the hand-checked point, H1 + fx1 on DS-120: ♭ is an IMMA.16832, 1/39 LDSM_X4 and fx1's 8 ALU and
8 IMAD warp instructions a unit, B_♭ with ε_skip's work is Lean's 17.097571, and a GPU-second of linears at a cycle a
unit is (1 − 5.860%) · 17.097571 s; QKᵀ counts at 1.25, and a block divides by its GPUs. A block where the datacenter
places PoUW's compiled Program is its tiles' `z` products, a linear's each at the tile's K, of the int8 products census/v1
and Lean's `h1Rows` count there, r·c·(2K + 32m) a tile (benchmarks/datacenter's test of that Program credits this figure
as the block's compute)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_proved_ticks", Path(__file__).parents[1] / "proved_ticks.py")
P = importlib.util.module_from_spec(spec)
sys.modules["pouw_proved_ticks"] = P
spec.loader.exec_module(P)

S = P.sheet()
#: 4096 int8 products a cycle on every SMSP of one GPU for one second at f_max
SECOND = 4096 * 4 * 188 * 2430e6


def test_the_sheet_is_h1_fx1_on_ds120():
    assert (S.family, S.vector, S.design, S.sms, S.fmax_mhz) == ("sm_120", "DS-120", "dither-c-peel-fx1", 188, 2430)
    assert P.SHEET_EPS == 0.0586


def test_b_flat_is_the_floor_programs_issue_bound_with_its_skippable_work():
    data = P.device(S.family)
    assert P.EB.prices(S.design, data, S.vector).mma == pytest.approx(1 + 1 / 39 + 8 + 8, abs=5e-7)
    assert P.unit_cycles(S) == pytest.approx(17.097571, abs=5e-7)     # Accounting.TS.Headline.Sm120.flat


def test_a_gpu_second_of_linears_is_one_minus_eps_of_b_flat_seconds():
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1.0) == pytest.approx(0.9414 * 17.097571,
                                                                                                 abs=1e-6)
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, 0.0, 1.0) == pytest.approx(17.097571, abs=1e-6)


def test_qk_counts_at_cohalfs_words_and_a_block_divides_by_its_gpus():
    one = P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1.0)
    assert P.proved_block_ticks({"mac:qk": SECOND}, S, P.SHEET_EPS, 1.0) == pytest.approx(1.25 * one)
    mixed = {"mac:linears": SECOND, "mac:lm_head": SECOND, "mac:qk": SECOND, "mac:pv": SECOND}
    assert P.proved_block_ticks(mixed, S, P.SHEET_EPS, 1.0) == pytest.approx(4.25 * one)
    assert P.proved_block_ticks({"mac:linears": SECOND}, S, P.SHEET_EPS, 1e-4, gpus=8) == pytest.approx(one * 1e4 / 8)


@pytest.mark.parametrize("census", [{"gate": 1.0}, {"mac:fp8": 1.0}, {"alu:fx1": 1.0}, {"mac:linears": -1.0}])
def test_work_outside_the_designs_products_is_refused(census):
    with pytest.raises(ValueError, match="LP over the union"):
        P.units(census, S)


def test_a_family_without_a_sheet_clock_is_refused():
    with pytest.raises(ValueError, match="FMAX_MHZ"):
        P.sheet("sm_90")


def test_a_block_of_the_compiled_scheme_is_its_tiles_z_products_as_linears_and_other_work_is_refused():
    from verity.ml.protocols.pouw.circuit import h1tile

    assert (h1tile.SCHEME, h1tile.FOLD) == (P.SCHEME, "fx1") and all("S8" in k for k in h1tile.KINDS)
    # benchmarks/datacenter's compiled s8 grid on its block, one 4 × 16 tile at K = 64 (test_datacenter's `work["a"]`):
    # census/v1's int8 products are z's and the lanes' segment dots', 4·16·(64 + 64 + 32·2), of which z's 4·16·64 credit
    grid = {P.Tile(4, 16, 64, 64): 1}
    block = {"cuda:gate": 9455977, "mac:int8": 4 * 16 * (64 + 64 + 32 * 2)}
    assert P.placed(block, grid) == {"mac:linears": 4096} and P.units(P.placed(block, grid), S) == 1
    assert P.proved_block_ticks(P.placed(block, grid), S, P.SHEET_EPS, 1e-3) == pytest.approx(
        0.9414 * 17.097571 / (4 * 188 * 2430e3), rel=1e-7)
    with pytest.raises(ValueError, match="isn't eps1-sm120-h1fx1's"):
        P.placed({"mac:int8": 1, "mac:bf16": 1}, grid)
    for wrong in ({"mac:int8": 4096}, {"mac:int8": 2 * block["mac:int8"]}, {}):
        with pytest.raises(ValueError, match="aren't its H1 tiles'"):
            P.placed(wrong, grid)


needs_h1 = pytest.mark.skipif(importlib.util.find_spec("verity.ml.protocols.pouw.circuit.segment") is None,
                              reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
ROWS = Path(__file__).resolve().parents[3] / "Security" / "Proofs" / "Accounting" / "Gates" / "Rows.lean"


def h1_rows() -> dict[tuple[int, int, int], int]:
    """Lean's `h1Rows` (Rows.lean): each H1 tile row's `dotMacs`, by (r, c, K)."""
    block = ROWS.read_text().split("\ndef h1Rows ", 1)[1]
    rows = re.findall(r"\((\d+), (\d+), (\d+), \(\d+, (\d+), 0, 0, \d+, \d+\)\)", block[:block.index("]\n\n")])
    return {(int(r), int(c), int(k)): int(macs) for r, c, k, macs in rows}


#: (r, c, K): z's products (the linears placed credits) and every product census/v1 and `h1Rows` count
TILE_ROWS = {(4, 8, 64): (2048, 6144), (4, 16, 64): (4096, 12288), (4, 8, 96): (3072, 8192)}


@needs_h1
@pytest.mark.parametrize("r,c,k", TILE_ROWS)
def test_a_tiles_products_are_lean_h1_rows_and_placed_credits_z_at_its_k(r, c, k):
    """census/v1 counts an H1 tile's z and its lanes' segment dots as int8 products, r·c·(2K + 32m), 3× z's at K = 64;
    `placed` credits z's r·c·K, at the tile's own K."""
    linears, products = TILE_ROWS[(r, c, k)]
    t = P.Tile(r, c, k, r * c)
    assert h1_rows()[(r, c, k)] == t.products == products and t.linears == linears == r * c * k
    assert P.placed({"mac:int8": products}, {t: 1}) == {"mac:linears": linears}
    assert P.placed({"mac:int8": 3 * products}, {t: 3}) == {"mac:linears": 3 * linears}


#: (r, c, K): every product census/v1 counts in the one-stream tile (`PouwH1Tile_v3`), the lanes' segment dots' alone
TILE_ROWS_V3 = {(4, 8, 64): 4096, (4, 16, 64): 8192, (4, 8, 96): 5120}


@needs_h1
@pytest.mark.parametrize("r,c,k", TILE_ROWS)
def test_a_one_stream_tiles_products_are_its_lanes_and_placed_credits_z_at_its_k(r, c, k):
    """v3's z is recombined from the lanes' folded words, so its tile has `h1Rows`' products less z's own, and `placed`
    credits the same z products, r·c·K, as v1's tile."""
    linears, products = TILE_ROWS[(r, c, k)]
    t = P.Tile(r, c, k, r * c, 3)
    assert t.products == TILE_ROWS_V3[(r, c, k)] == h1_rows()[(r, c, k)] - linears and t.linears == linears
    assert P.placed({"mac:int8": t.products}, {t: 1}) == {"mac:linears": linears}
    with pytest.raises(ValueError, match="aren't its H1 tiles'"):
        P.placed({"mac:int8": products}, {t: 1})


def test_lean_h1_rows_are_the_three_rows():
    assert set(h1_rows()) == set(TILE_ROWS)


@needs_h1
@pytest.mark.slow
@pytest.mark.parametrize("version", [1, 3])
@pytest.mark.parametrize("r,c,k", TILE_ROWS)
def test_placed_on_a_tiles_program_reads_census_v1(r, c, k, version):
    """The tile's Program, read as the datacenter reads a block (`logical.work`'s `mac:int8`, here census/v1's products by
    class), on v1 and on the one stream v3, is `tiles`' one tile and `placed` credits z's products. Slow on a process's first tile: census/v1 traces its
    seeds' SHA-512 on bits (about 35 s), then about 1 s a row; the fast rows above take the products from Lean alone."""
    from verity.ml.protocols.pouw.circuit import h1tile as H
    from verity.ml.circuits.census import PRODUCTS, census
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.program import Program

    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    defn = {1: H.PouwH1Tile, 3: H.PouwH1Tile3}[version]
    program = Program(bind(defn, EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c))))
    doc = census(program)
    int8 = sum(cl["instances"] * PRODUCTS.get(cl["family"], 0) for cl in doc["classes"].values())
    assert P.tiles(program) == {P.Tile(r, c, k, r * c, version): 1}
    assert int8 == (TILE_ROWS[(r, c, k)][1] if version == 1 else TILE_ROWS_V3[(r, c, k)]) and doc["totals"]["served_macs"] == doc["totals"]["tail_atoms"] == 0
    assert P.placed({"mac:int8": int8}, P.tiles(program)) == {"mac:linears": TILE_ROWS[(r, c, k)][0]}
