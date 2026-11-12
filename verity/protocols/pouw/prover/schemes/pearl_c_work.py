"""Pearl-C's records: what the prover declares for each matmul before the samples, and the root that binds them (statement
review item 5).  vLLM's Pearl-C prover writes them (`verity_vllm.protocols.pouw.pearl_c`).  The work sampling design
that credits, samples and checks by them, with its clear audit, is a reference tool, not a verifier:
``verity/protocols/pouw/clear/clear_work.py``.

* **`Split`**: a weight's channel split.
  - S is a set of at most ⌊k/100⌋ input channels, computed outside the unit.
  - The unit reads the |K| = k − |S| kept channels in order, zero-padded to `k_unit`, a multiple of 128.
  - `calibration` names the data S was computed from. The pinned rule that computes S (`split_channels`, with its
    tie-break and budget) lives with the model integration.
* **`UnitRecord`**: what the prover declares for each matmul before the samples.
  - Its real rows. The rest are filler, which must be zero words.
  - Its kept width |K|.
  - Its excluded rows: A rows of the matmul and B rows of its weight that fail the scheme's per-row rules.
  - Its voluntary rows (V-EX): rows it gives up, which earn no credit.
  - The band rule (`UnitRecord.credited`): a voluntary A row takes the credit of its whole band of 64 A rows, and an
    excluded or voluntary B row takes the credit of its whole block of 64 columns. The ticket weight, the inert-work bound
    and the credit all read it.
  `records_root` binds the records and the splits.
* **`pad_weight`**: a weight whose n is not whole column tiles (GLM-5.3's lm_head, 19360 rows a TP8 rank) is served with
  zero rows up to the next whole tile, and the last tile's rows are voluntary, so their block is neither credited nor sampled.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.commitments import domains

from verity.protocols.pouw.verifier.protocol import Matmul, Tile, tagged

RECORDS_TAG = domains.POUW_PEARL_C_RECORDS
SPLIT_TAG = domains.POUW_PEARL_C_SPLIT
ALIGN = 128
#: The band rule's band of A rows and block of B rows, at absolute indices: the game's 64 × 64 tiling.
BAND = 64


@dataclass(frozen=True)
class Split:
    """One weight's channel split: the layer's k input channels, the side set S, and the calibration data's digest."""

    k: int
    side: tuple[int, ...] = ()
    calibration: bytes = b""

    def __post_init__(self) -> None:
        if any(not 0 <= c < self.k for c in self.side) or any(a >= b for a, b in zip(self.side, self.side[1:])):
            raise ValueError("S is strictly increasing channels of the layer")
        if len(self.side) > self.k // 100:
            raise ValueError(f"|S| = {len(self.side)} exceeds 1% of k = {self.k}")

    @property
    def k_kept(self) -> int:
        return self.k - len(self.side)

    @property
    def k_unit(self) -> int:
        return -(-self.k_kept // ALIGN) * ALIGN

    def encode(self) -> bytes:
        return struct.pack(f"<II{len(self.side)}I", self.k, len(self.side), *self.side) + tagged(SPLIT_TAG, self.calibration)


@dataclass(frozen=True)
class UnitRecord:
    """One matmul as its prover declares it before the samples."""

    matmul: Matmul                  # index, (m, k_unit, n), weight
    real_rows: int                  # rows 0 … real_rows − 1 are the call's; the rest are zero filler
    k_kept: int                     # entries 0 … k_kept − 1 of every row are the kept channels; the rest are zero
    excluded_a: tuple[int, ...] = ()  # A rows (< real_rows) that fail the per-row rules
    excluded_b: tuple[int, ...] = ()  # B rows of the weight that fail them
    voluntary_a: tuple[int, ...] = ()  # A rows (< real_rows) given up before the samples (V-EX)
    voluntary_b: tuple[int, ...] = ()  # B rows of the weight given up (V-EX)

    def __post_init__(self) -> None:
        s = self.matmul.shape
        if not (0 < self.real_rows <= s.m and 0 < self.k_kept <= s.k):
            raise ValueError("a record's real rows and kept width lie within its shape")
        for rows, bound in ((self.excluded_a, self.real_rows), (self.excluded_b, s.n), (self.voluntary_a, self.real_rows),
                            (self.voluntary_b, s.n)):
            if any(not 0 <= r < bound for r in rows) or any(a >= b for a, b in zip(rows, rows[1:])):
                raise ValueError("excluded and voluntary rows are strictly increasing and in range")
        if set(self.excluded_a) & set(self.voluntary_a) or set(self.excluded_b) & set(self.voluntary_b):
            raise ValueError("a row is excluded or voluntary, not both")

    def encode(self) -> bytes:
        """The v2 layout; a record with voluntary rows appends their counts and rows (each record is one framed part of
        `records_root`, so the two layouts can't be confused)."""
        u, s = self.matmul, self.matmul.shape
        a, b = self.excluded_a, self.excluded_b
        base = struct.pack(f"<QQIIIIIII{len(a)}I{len(b)}I", u.index, u.weight, s.m, s.k, s.n, self.real_rows,
                           self.k_kept, len(a), len(b), *a, *b)
        va, vb = self.voluntary_a, self.voluntary_b
        return base + struct.pack(f"<II{len(va)}I{len(vb)}I", len(va), len(vb), *va, *vb) if va or vb else base

    def credited(self, tile: Tile) -> tuple[list[int], list[int]]:
        """The tile's credited rows and columns, by the band rule:
        - an A row is credited when it is real, not excluded, and its band of `BAND` rows (``i // BAND``) holds no
          voluntary A row;
        - a column is credited when its block of `BAND` B rows holds no excluded or voluntary B row.
        At the deployed 64 × 64 tile, a tile holding a voluntary row, or an excluded B row, has no credited cell, and on
        every other tile the credited rows × columns are the game's passing rows × all its columns (`auditTiling 64 64`)."""
        bands = {i // BAND for i in self.voluntary_a}
        blocks = {j // BAND for j in (*self.excluded_b, *self.voluntary_b)}
        ea = set(self.excluded_a)
        return ([i for i in tile.rows if i < self.real_rows and i not in ea and i // BAND not in bands],
                [j for j in tile.cols if j // BAND not in blocks])


def pad_weight(rows: Sequence[bytes], tile_n: int) -> tuple[list[bytes], tuple[int, ...]]:
    """(the rows, the padded tile's rows): a weight's rows (FP32 words along k) with all-zero rows appended up to a whole
    number of `tile_n`-column tiles, and the rows of the last tile when it holds any of them, its real columns with the
    zero rows.  The glue drops the padded rows' outputs.  The prover declares the padded tile's rows voluntary
    (`voluntary_b`, through `declare`'s `padded_b`), so by the band rule their block of 64 columns has no credited cell:
    no sample lands on it and it opens no exclusion tile, so no tile check sees a zero row, and the credit is the other
    blocks' columns at the padded n's cell credit, under an unpadded n's honest credit."""
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("a weight's rows are one length")
    pad = -len(rows) % tile_n
    return list(rows) + [bytes(len(rows[0]))] * pad, tuple(range(len(rows) // tile_n * tile_n, len(rows) + pad) if pad else ())


def records_root(salt: bytes, weights_root: bytes, splits: Mapping[int, Split], records: Sequence[UnitRecord]) -> bytes:
    return tagged(RECORDS_TAG, salt, weights_root, *(p for w in sorted(splits) for p in (w, splits[w].encode())),
                  *(r.encode() for r in records))
