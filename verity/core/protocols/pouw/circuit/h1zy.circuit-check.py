"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.h1zy`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwZyCheck_v1` at the smoke's 4 × 8 unit (32 `PouwZyDequant_v2` and `PouwZyCell_v2`, then 31 `PouwAndBit_v2`), and
    `PouwH1TileZy_v1` over `PouwH1Tile_v1` and `PouwH1Tile_v3`, and over J10-R's `PouwH1Tile_v5` and `PouwH1Tile_v6` (whose
    call and blocks it passes on), at their smallest tile, one lane (4 × 8) at K = 64 on the s8 accumulator on bits
    (`h1tile`'s sidecar's), each the tile and that check. `PouwH1LaneZy_v1` over both at the smallest
    leaf of whole index bits, 8 × 16 (4 lanes, the low 2 bits of a 16-bit index): one lane's seed, 13 noise blocks and its
    key block, each counter picked by `PouwMuxBit_v2` over the index, its fx1 chain, z (v1's exact dots or v3's one-stream
    cells), its 32 record cells picked over the index, that check, the admit gate and the equality of its partials row (8
    words at KS = 2), and ok 0 where the index's 14 high bits are not all 0;
    and `PouwH1Combine_v1` at that leaf, its 4 rows XORed and held to the receipt. `PouwH1LaneZyChunk_v1` over
    `PouwH1Tile_v1` at that leaf: the first and last of 2 chunks at K = 64 and the middle of 3 at K = 96 (one 32-entry
    segment a chunk), the first with its key block and `GemmAccumulatorS8_v2` at the chunk's K, the others continuing the
    carried chain and z, each its carry out's equality, the last the record's check and the partials' (the whole lane's
    8 and 10 words). `PouwH1TileZyChunks_v1` over the v1 unit at K = 64 on 2 chunk rows a row: each whole row joined from
    its chunk rows, the unit on them, and each later chunk's scale held to chunk 0's (12 `PouwEq32_v2`), and
    `PouwH1TileZyChunks_v2` the same over J10-R's v5 and v6 units, passing their call and blocks on. J10-R's lanes over
    `PouwH1Tile_v5` at that leaf: `PouwH1LaneZy_v2`, v1's lane with its seed_c and σ_R from one `PouwCallSeeds_v2` and each
    block's counter the picked offset plus the call's blocks (`PouwAdd64_v2`), `PouwH1LaneZyChunk_v2` at v1's three
    chunks, and `PouwH1Combine_v2` over the v5 leaf, v1's receipt check AND its 4 lanes' ok rows (each the word 1). All
    are bound as
    Definitions, since `h1zy.lane`, `h1zy.lane_chunk` and `h1zy.combine` stage only a
    leaf whose R is a multiple of 16, and the combine only at K >= 384 (`h1zy.LEAST_K`). None where
    `circuit.segment` is not on the tree, which is where `PouwH1Tile` is registered. `PouwZyDequant`, `PouwZyCell`,
    `PouwAndBit` and `PouwMuxBit` are catalog roots (static-free)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import h1zy as Z

    if H.SG is None:
        return []
    def tile(d: Any, r: int, c: int, k: int = 64) -> Any:
        exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
        return bind(d, EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c)))

    tiles = (H.PouwH1Tile, H.PouwH1Tile3)
    return ([bind(Z.PouwZyCheck, R=4, C=8)] + [Z.unit(tile(d, 4, 8)) for d in (*tiles, H.PouwH1Tile5, H.PouwH1Tile6)]
            + [Z.unit_chunks(tile(d, 4, 8), 2) for d in (H.PouwH1Tile, H.PouwH1Tile5, H.PouwH1Tile6)]
            + [bind(Z.PouwH1LaneZy, TILE=tile(d, 8, 16)) for d in tiles]
            + [bind(Z.PouwH1Combine, TILE=tile(H.PouwH1Tile, 8, 16))]
            + [bind(Z.PouwH1LaneZyChunk, TILE=tile(H.PouwH1Tile, 8, 16, k), CHUNKS=n, CHUNK=c)
               for k, n, c in ((64, 2, 0), (64, 2, 1), (96, 3, 1))]
            + [bind(Z.PouwH1LaneZy2, TILE=tile(H.PouwH1Tile5, 8, 16)), bind(Z.PouwH1Combine2, TILE=tile(H.PouwH1Tile5, 8, 16))]
            + [bind(Z.PouwH1LaneZyChunk2, TILE=tile(H.PouwH1Tile5, 8, 16, k), CHUNKS=n, CHUNK=c)
               for k, n, c in ((64, 2, 0), (64, 2, 1), (96, 3, 1))])
