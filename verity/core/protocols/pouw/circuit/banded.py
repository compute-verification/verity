"""J10's banded H1 noise on words: `h1tile`'s unit under one noise seed a call, the bit-exact reference a banded kernel is
held to. The luck claims cover it at the same values; it is served only once kernels measure the gain. It is not the
unit's gates. Its coin is the Lean `Accounting.TS.CallCoin` of one row block by one
band.

A call's column tiles are cut into bands of g ≤ `BAND_MAX` column tiles from column tile 0 (`band_start`), the last band
holding what remains. Tile (I, J), at unit position ``positions[I][J]``, reads:

* its rows' noise (`h1tile.TileShape.row`'s blocks: dither bits, digit coins) from `seeds.seed` at the position of its band's
  first tile, (I, J - J mod g), so every tile of a band reads the same rows and other bands and row blocks read other
  seeds;
* its columns' noise (`h1tile.TileShape.col`'s blocks: coin bits, U) from `seeds.seed` at its own position;
* its fold keys from `seeds.receipt_seed` at its own position (`h1tile.fold_keys`).

At g = 1 every tile reads its own seed only: `h1tile.noise` and `h1tile.fold_keys`, and `banded_reference` is
`h1tile.reference`. The band's rows read counter blocks [0, r·RB) of the band's seed and each column reads blocks from r·RB
on of its own tile's seed, so no noise block serves two purposes.
"""
from __future__ import annotations

from typing import Sequence

from . import h1tile as H
from . import seeds as SD

#: the widest band `Accounting.Guarantees.TS.GridTemplateMoment` prices at the one-tile rate, in column tiles
BAND_MAX = 64


def band_start(J: int, g: int) -> int:
    """The first column tile of tile column J's band of g."""
    if not 1 <= g <= BAND_MAX:
        raise ValueError(f"a band is 1 to {BAND_MAX} column tiles, not {g}")
    if J < 0:
        raise ValueError(f"a column tile index is >= 0, not {J}")
    return J - J % g


def row_noise(seed_j: bytes, lay: H.TileShape) -> list:
    """seed_j's row noise, `h1tile.noise`'s rows: per row of the tile, (dither bits, digit coins)."""
    k, m = lay.k, lay.m
    rows = []
    for i in range(lay.r):
        s = H._stream(seed_j, lay.row(i), lay.row_bits)
        rows.append((s[:k], [s[k + 32 * d:k + 32 * d + 32] for d in range(m - 1)]))
    return rows


def col_noise(seed_j: bytes, lay: H.TileShape) -> list:
    """seed_j's column noise, `h1tile.noise`'s columns: per column of the tile, (coin bits, U's int8 encodings)."""
    k = lay.k
    cols = []
    for j in range(lay.c):
        s = H._stream(seed_j, lay.col(j), lay.col_bits)
        cols.append((s[:k], [H.u129(H._value(s[k + 8 * t:k + 8 * t + 8])) for t in range(32)]))
    return cols


def banded_noise(salt: bytes, positions: Sequence[Sequence[int]], roots: tuple[bytes, bytes], lay: H.TileShape, I: int,
                 J: int, g: int) -> tuple[list, list, list[list[int]]]:
    """Tile (I, J)'s (rows, cols, keys) banded at g: `row_noise` of its band's seed, `col_noise` of its own and
    `h1tile.fold_keys` of its own σ_R. `positions[I][J]` is tile (I, J)'s unit position (`compile.Compiled.positions`) and
    `roots` the call's seed group's."""
    J0 = band_start(J, g)
    if not 0 <= I < len(positions) or not J < len(positions[I]):
        raise ValueError(f"tile ({I}, {J}) is outside the call's grid of positions")
    own = positions[I][J]
    return (row_noise(SD.seed(salt, positions[I][J0], *roots), lay), col_noise(SD.seed(salt, own, *roots), lay),
            H.fold_keys(SD.receipt_seed(salt, own, *roots), lay))


def banded_reference(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], salt: bytes,
                     positions: Sequence[Sequence[int]], roots: tuple[bytes, bytes], I: int, J: int, g: int,
                     pairs: Sequence[tuple[int, int]] | None = None) -> tuple[list[int], list[int]]:
    """Tile (I, J)'s unit on words banded at g: `h1tile.reference`'s (z, receipt), its sets and fold the same, with its
    noise and keys from `banded_noise`. z is each cell's A Bᵀ, which no noise reaches."""
    SG = H.SG
    lay = H.TileShape(len(a), len(b), len(a[0]))
    rows, cols, keys = banded_noise(salt, positions, roots, lay, I, J, g)
    cp = [[0, 0] for _ in range(lay.sets)]
    for ln in range(lay.lanes):
        R, Cs = lay.rows(ln), lay.cols(ln)
        seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
        xr = [[[a[i][t] for t in ts] for i in R] for ts in seg]
        bb = [[[rows[i][0][t] for t in ts] for i in R] for ts in seg]
        wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
        cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
        st = list(keys[ln])
        for s, p in enumerate(SG.call_words(xr, bb, wr, cc, [cols[j][1] for j in Cs],
                                            [[rows[i][1][d] for i in R] for d in range(lay.m - 1)])):
            st = SG.absorb(H.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
    if pairs is None:
        pairs = [(i, j) for i in range(lay.r) for j in range(lay.c)]
    return [SG.seg_dot(0, a[i], b[j]) for i, j in pairs], [w & H.M32 for s in cp for w in s]


__all__ = ["BAND_MAX", "band_start", "banded_noise", "banded_reference", "col_noise", "row_noise"]
