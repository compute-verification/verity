"""The unpeeled one-stream unit of PouwH1Tile_v3 with the exact wrap correction, on words, for the dither gates:
`reference(..., stream=True)`'s z is each cell's O' - sum_d 128^d D_d plus 256 X (wb+ - wb-)^T mod 2^32, X (B + c)^T (v3
before the peel, at 5f54fbe355), and the receipt (v1's). The word functions are h1tile's own (wrap_bits,
wrap_seg, one_stream, lane_sets, col_wrap: v3's, and v4's); only the unpeeled `stream_cell` and `reference`, which h1tile
now states peeled (z = A B^T, `peel_seg`), are kept here.
"""
from __future__ import annotations

from typing import Sequence

from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG
from verity.core.protocols.pouw.circuit.h1tile import (FOLD, M32, Layout, col_wrap, fold_keys, lane_sets, noise,  # noqa: F401
                                                       one_stream, wrap_bits, wrap_seg)

SOURCE = "h1tile's word functions; the unpeeled stream_cell and reference at 5f54fbe355"


def stream_cell(op: int, d: Sequence[int], x_i: Sequence[int], wb_j: Sequence[int]) -> int:
    """A cell's v3 z: `one_stream` of its O' and D words, then `wrap_seg` over each segment of its row's X and its column's
    wrap bits; X Ŵᵀ mod 2^32 for every B."""
    z = one_stream(op, d)
    for s in range(0, len(x_i), 32):
        z = wrap_seg(z, x_i[s:s + 32], wb_j[s:s + 32])
    return z


def reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes, position: int,
              roots: tuple[bytes, bytes], pairs: Sequence[tuple[int, int]] | None = None,
              stream: bool = False) -> tuple[list[int], list[int]]:
    """The unit on words: (z, receipt) for A's r rows and B's c rows (int8 encodings), its salt, position and roots, z at
    `pairs` (the full grid in row-major order by default). `stream`: v3's z, each cell's `one_stream` of its lane's O' and
    D words with its wrap correction (`stream_cell`); else v1's, the exact A Bᵀ cell."""
    lay = Layout(len(a), len(b), len(a[0]))
    rows, cols = noise(SD.seed(salt, position, *roots), lay)
    xs = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)] if stream else None
    wbs = [col_wrap(b[j], cols[j]) for j in range(lay.c)] if stream else None
    keys = fold_keys(SD.receipt_seed(salt, position, *roots), lay)
    ks = lay.ks
    cp = [[0, 0] for _ in range(lay.sets)]
    served: dict[tuple[int, int], int] = {}
    for ln in range(lay.lanes):
        sets = lane_sets(a, b, rows, cols, lay, ln)
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
        for e in range(32 if stream else 0):
            cell = (lay.rows(ln)[SG.row_slot(e)], lay.cols(ln)[SG.col_slot(e)])
            served[cell] = stream_cell(sets[ks][e], [d[e] for d in sets[ks + 1:]], xs[cell[0]], wbs[cell[1]])
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    z = [served[p] for p in pairs] if stream else [SG.seg_dot(0, a[i], b[j]) for i, j in pairs]
    return z, [w & M32 for s in cp for w in s]
