"""The layout of the FA2 *hidden* value stream (scores, P, running max, rescale, lane sums, PV accumulators, epilogue)
as the v7 taps write it.

The kernel materialises the selected classes into one flat u32 buffer (`VERITY_TAP=3`, the M1 stream) whose layout is
arithmetic in (HB = b*h slabs, M = seqlen_q rows, NB = key-block cap): `StreamLayout` below mirrors `Verity_mat_layout` in
`verity_tap.h`, and every canonical (class, slab, row, col | block, lane | d) position is one word.  The tap exactness
properties and the hidden source's tests place a word by it; no library module reads it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verity_vllm.commit import scheme

__all__ = ["ROW_GUARD_FIELD", "SRC_BITS", "StreamLayout"]

VERSION = scheme.FA2H_VERSION
KBLOCK_N = 128
#: MS (64): the seventh class, appended after FIN only by the guarded-max tap builds (`VERITY_MAT_SRC & 64`, `hidden_source.TAP_ROW3`): one f32
#: per (slab, row, key block), the `max * scale` that block's exp2 FMAs read.  Without it the layout, digest and bytes are the six classes'.
SRC_BITS = {"S": 1, "P": 2, "RP": 4, "ROW": 8, "O": 16, "FIN": 32, "MS": 64}
#: ROW word 3: never written, except by the guarded-max tap builds (`verity_tap_row3() == 1`, `hidden_source.TAP_ROW3`), which store
#: GuardNegInfZero(row_max) there at every key block after a row's first and 0 at the first
ROW_GUARD_FIELD = 3


@dataclass(frozen=True)
class StreamLayout:
    """Word offsets of the M1 stream (mirror of `Verity_mat_layout(HB, M, NB, BN, D)`).

    Tap layout v8: the layout is arithmetic in the kernel tile -- `BN` = kBlockN of the launched kernel (128 for head
    dim 64, 64 for 96/128/256 on the num_splits == 1 kernel; `hidden_source.fa2_kblock_n`) and `D` = head_dim (the O plane's row
    width).  The defaults reproduce the legacy hd-64 layout, digest and chunk header word for word; other tiles enter the digest and the
    spare header words 8/9, so a chunk of a (64, 128) stream can never verify against a (128, 64) declaration.  `ms` appends the MS
    class after FIN (and enters the digest); without it every offset, the total and the digest are the six classes'."""

    HB: int
    M: int
    NB: int
    BN: int = KBLOCK_N
    D: int = 64
    ms: bool = False

    @property
    def N(self) -> int:
        return self.NB * self.BN

    @property
    def offsets(self) -> dict[str, int]:
        HB, M, N, NB, D = self.HB, self.M, self.N, self.NB, self.D
        off_s = 0
        off_p = off_s + HB * M * N
        off_rp = off_p + HB * M * N
        off_row = off_rp + HB * M * (N // 2)
        off_o = off_row + HB * M * NB * 8
        off_fin = off_o + HB * M * NB * D
        o = {"S": off_s, "P": off_p, "RP": off_rp, "ROW": off_row, "O": off_o, "FIN": off_fin, "total": off_fin + HB * M * 4}
        if self.ms:
            o["MS"] = o["total"]
            o["total"] = o["MS"] + HB * M * NB
        return o

    @property
    def legacy_tile(self) -> bool:
        return (self.BN, self.D) == (KBLOCK_N, 64)

    @property
    def total_words(self) -> int:
        return self.offsets["total"]

    def padded_words(self, chunk_words: int = 64) -> int:
        """Stream length rounded up to whole chunks (the tail chunk is zero-padded; padding words are outside the layout)."""
        return -(-self.total_words // chunk_words) * chunk_words

    def word_index(self, cls: str, slab: int, row: int, *, col: int | None = None, block: int | None = None,
                   lane: int | None = None, d: int | None = None, field: int | None = None) -> int:
        """Canonical position -> word index.  ROW fields: 0 row_max, 1 rescale, 2 step, 3 the guarded max (`ROW_GUARD_FIELD`),
        4+lane l_lane.  FIN: 0 sum, 1 inv_sum, 2 final row_max, 3 lse.  MS (with `ms`): one word per block."""
        o = self.offsets
        if not (0 <= slab < self.HB and 0 <= row < self.M):
            raise ValueError("position outside the layout")
        place = _PLACE_IN_ROW.get(cls)
        if place is None or cls not in o:                # an unknown class, or MS on a layout without it
            raise ValueError(cls)
        words, at = place(self, col, block, lane, d, field)
        return o[cls] + (slab * self.M + row) * words + at

    def unrank(self, word: int) -> dict[str, Any]:
        o = self.offsets
        if word >= o["total"] or word < 0:
            raise ValueError("word outside the layout")
        if word < o["P"]:
            i = word - o["S"]
            sr, col = divmod(i, self.N)
            slab, row = divmod(sr, self.M)
            return {"cls": "S", "slab": slab, "row": row, "col": col}
        if word < o["RP"]:
            i = word - o["P"]
            sr, col = divmod(i, self.N)
            slab, row = divmod(sr, self.M)
            return {"cls": "P", "slab": slab, "row": row, "col": col}
        if word < o["ROW"]:
            i = word - o["RP"]
            sr, cw = divmod(i, self.N // 2)
            slab, row = divmod(sr, self.M)
            return {"cls": "RP", "slab": slab, "row": row, "col": 2 * cw}
        if word < o["O"]:
            i = word - o["ROW"]
            srb, f = divmod(i, 8)
            sr, block = divmod(srb, self.NB)
            slab, row = divmod(sr, self.M)
            return {"cls": "ROW", "slab": slab, "row": row, "block": block, "field": f}
        if word < o["FIN"]:
            i = word - o["O"]
            srb, d = divmod(i, self.D)
            sr, block = divmod(srb, self.NB)
            slab, row = divmod(sr, self.M)
            return {"cls": "O", "slab": slab, "row": row, "block": block, "d": d}
        if self.ms and word >= o["MS"]:
            sr, block = divmod(word - o["MS"], self.NB)
            slab, row = divmod(sr, self.M)
            return {"cls": "MS", "slab": slab, "row": row, "block": block}
        i = word - o["FIN"]
        sr, f = divmod(i, 4)
        slab, row = divmod(sr, self.M)
        return {"cls": "FIN", "slab": slab, "row": row, "field": f}

    def digest(self) -> bytes:
        d = {"kind": "fa2-hidden-m1", "v": VERSION, "HB": self.HB, "M": self.M, "NB": self.NB}
        if not self.legacy_tile:
            d.update({"BN": self.BN, "D": self.D})
        if self.ms:
            d["ms"] = True
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).digest()


# One function a class: (the words one (slab, row) takes in the class, the position's word among them).
def _col(L: StreamLayout, col: int | None, *_: int | None) -> tuple[int, int]:
    assert col is not None and 0 <= col < L.N
    return L.N, col


def _col_pair(L: StreamLayout, col: int | None, *_: int | None) -> tuple[int, int]:
    assert col is not None and 0 <= col < L.N
    return L.N // 2, col // 2                            # word holds cols (c, c+1); bf16 halves


def _row_field(L: StreamLayout, _col: int | None, block: int | None, lane: int | None, _d: int | None,
               field: int | None) -> tuple[int, int]:
    assert block is not None and 0 <= block < L.NB
    if field is None:
        assert lane is not None
        field = 4 + lane
    return L.NB * 8, block * 8 + field


def _output(L: StreamLayout, _col: int | None, block: int | None, _lane: int | None, d: int | None,
            *_: int | None) -> tuple[int, int]:
    assert block is not None and d is not None and 0 <= d < L.D
    return L.NB * L.D, block * L.D + d


def _final(L: StreamLayout, _col: int | None, _block: int | None, _lane: int | None, _d: int | None,
           field: int | None) -> tuple[int, int]:
    assert field is not None
    return 4, field


def _block(L: StreamLayout, _col: int | None, block: int | None, *_: int | None) -> tuple[int, int]:
    assert block is not None and 0 <= block < L.NB
    return L.NB, block


_PLACE_IN_ROW: dict[str, Callable[..., tuple[int, int]]] = {
    "S": _col, "P": _col, "RP": _col_pair, "ROW": _row_field, "O": _output, "FIN": _final, "MS": _block}
