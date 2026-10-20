"""A CPU twin of CA's one-stream tile, PouwH1Tile_v3 with the exact wrap correction, for this lane's gates: the word-level
functions below are copied verbatim from verity/core/protocols/pouw/circuit/h1tile.py on cursor/h1-v3-1172 at 5f54fbe355
(H1V3's owner, 08:59Z), which isn't on main yet. They run on main's h1tile (Layout, noise, fold_keys, FOLD: v3's are
main's, bit for bit), seeds and segment. `reference(..., stream=True)` is v3's unit on words: each cell's z =
O' - sum_d 128^d D_d plus 256 X (wb+ - wb-)^T mod 2^32, and the receipt (v1's). Replace this file by an import once v3
lands.
"""
from __future__ import annotations

from typing import Sequence

from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG
from verity.core.protocols.pouw.circuit.h1tile import FOLD, M32, Layout, fold_keys, noise

SOURCE = "verity/core/protocols/pouw/circuit/h1tile.py @ cursor/h1-v3-1172 5f54fbe355"


def _sx(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def wrap_bits(b: int, c: int, u: int) -> int:
    """W' = B + c + U's int8 wrap at one entry: 1 (wb⁺) where B + c + U > 127, 2 (wb⁻) where it is < -128, else 0."""
    s = _sx(b) + (c & 1) + _sx(u)
    return 1 if s > 127 else 2 if s < -128 else 0


def wrap_seg(acc: int, x: Sequence[int], wb: Sequence[int]) -> int:
    """acc + 256 sum_e x_e (wb⁺_e - wb⁻_e) mod 2^32 over one segment: the wrap correction of a cell's one stream, x the int8
    encodings of X and wb `wrap_bits`' (both bits set: 0), so a flipped bit moves the sum by ±256 x_e."""
    return (acc + (sum(_sx(xe) * ((w & 1) - (w >> 1 & 1)) for xe, w in zip(x, wb)) << 8)) & M32


def one_stream(op: int, d: Sequence[int]) -> int:
    """cgemm.cu's served output from a cell's absorbed words: O' - sum_d 128^d D_d mod 2^32 (d = 1 .. m - 1)."""
    return (op - sum(w << 7 * (t + 1) for t, w in enumerate(d))) & M32


def lane_sets(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], rows: list, cols: list, lay: Layout,
              ln: int) -> list[list[int]]:
    """Lane ln's absorbed sets on words (`segment.call_words`): T_0 .. T_(KS-1), O', D_1 .. D_(m-1), 32 entries each."""
    R, Cs = lay.rows(ln), lay.cols(ln)
    seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
    xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
    bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
    wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
    cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
    return SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs], [[rows[i][1][d] for i in R] for d in range(lay.m - 1)])


def col_wrap(b_j: Sequence[int], col: tuple) -> list[int]:
    """Column j's wrap bits at every entry of K (`wrap_bits` of B, its coin bit and U, U repeating every 32 entries)."""
    return [wrap_bits(v, col[0][t], col[1][t % 32]) for t, v in enumerate(b_j)]


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
