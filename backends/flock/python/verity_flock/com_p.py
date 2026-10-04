"""com_P (``rec/com-p/v0``): a hidden unit circuit's nonzeros as a registered value, the recursive session's R0.

A hidden unit circuit is its ``A_0`` and ``B_0`` inside a unit slot of ``2^SL`` rows (C-Flock's ``flock-ir-unit/v2``
netlist: row ``i`` reads the columns ``rows_a[i]`` and ``rows_b[i]``). com_P holds it as rows of whole SHA-512 blocks
(`verity.ml.boolean.comb`: `Shape`, `pack`): per bucket ``s < 64`` (the column's low six bits) its R rows of E entries
(side, row, column >> 6; W = 2 SL - 4 bits, LSB first), zero-padded, then the column ``COL``'s A bits and B bits. The shape
(SL, R, COL, blocks) is the class's, fixed before any hidden circuit: it is the public bound on the nonzeros, so a class
member's matrices reveal nothing past it.

**Registration.** R0 registers the rows as the class's registered value `VALUE`:
``verity_sampled_proofs.one_stage.registered.Registered.commit(cls, VALUE, rows)``, each row's ``hm96-sha512/row/v1`` leaf
under a fresh 192-byte salt, the frame-v3-sha512 root under the domain of ``(cls, VALUE, words, leaves)``. ``cls`` names the
class, coin-free: never the hidden statement's digest (a function of the matrices) nor V*'s (a function of the coins).
V*'s chunk unit ``k`` reads row ``k`` at public position ``k`` (a registered port), so the rows it evaluates are the
registered ones.

**The statement.** `hidden_entries` is the map to the hidden statement's entry list (rec-lean's ``Hidden``: ``(side,
row, column, value)``, side 0 for A and 1 for B, in canonical form, padded to the shape's `Shape.nnz`); `matrices` gives the
same matrices as rows of columns.
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.boolean import comb as K

FORMAT = "rec/com-p/v0"
VALUE = "com-p"


def shape(SL: int, bound: int, COL: int | None, blocks: int = 1) -> K.Shape:
    """The fewest rows a bucket that hold ``bound`` entries."""
    E = K.per_row(SL, blocks)
    return K.Shape(SL, -(-bound // E), COL, blocks)


def rows(sh: K.Shape, rows_a: Sequence[Sequence[int]], rows_b: Sequence[Sequence[int]]) -> list[list[int]]:
    """com_P's rows for the unit circuit's matrices (`pack` of its `entries`; a bucket over the bound is refused)."""
    buckets, a, b = K.entries(rows_a, rows_b, sh.SL, sh.COL)
    return K.pack(sh, buckets, a, b)


def _cells(sh: K.Shape, com: Sequence[Sequence[int]]) -> dict[tuple[int, int, int], int]:
    buckets, a, b = K.unpack(sh, com)
    cells: dict[tuple[int, int, int], int] = {}
    for s, es in enumerate(buckets):
        for e in es:
            side, i, u = K.decode(sh.SL, e)
            for k in (0, 1):
                if side >> k & 1:
                    key = (k, u << 6 | s, i)
                    cells[key] = cells.get(key, 0) ^ 1
    if sh.COL is not None:
        for k, bits in ((0, a), (1, b)):
            for i, x in enumerate(bits):
                if x:
                    key = (k, sh.COL, i)
                    cells[key] = cells.get(key, 0) ^ 1
    return cells


def hidden_entries(sh: K.Shape, com: Sequence[Sequence[int]]) -> list[tuple[int, int, int, int]]:
    """The statement's entries: the value-1 cells strictly increasing in (side, column, row), then ``(0, 0, 0, 0)`` up to
    `Shape.nnz` (equal cells of com_P cancel, as in the matrices)."""
    ones = sorted(((k, i, c, 1) for (k, c, i), v in _cells(sh, com).items() if v), key=lambda t: (t[0], t[2], t[1]))
    return ones + [(0, 0, 0, 0)] * (sh.nnz - len(ones))


def matrices(sh: K.Shape, com: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """The matrices com_P's rows hold: each row's A and B columns, ascending."""
    n = 1 << sh.SL
    ra: list[list[int]] = [[] for _ in range(n)]
    rb: list[list[int]] = [[] for _ in range(n)]
    for (k, c, i), v in sorted(_cells(sh, com).items(), key=lambda kv: (kv[0][2], kv[0][1])):
        if v:
            (ra if k == 0 else rb)[i].append(c)
    return ra, rb
