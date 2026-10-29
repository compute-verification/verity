"""H1's served rows as their ``hm96-sha512/row/v2`` bit rows' u16 words (`pouw_h1_served`'s tables): a call's A rows
(`row_words`, as K's chunks `chunk_words`), a step's input ids (`ids_words`), a call's receipts (`receipt_words`) and its
z/y records (`record_words`, `zy_words`)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from verity_vllm.protocols.pouw.h1 import records as ZR

#: an input-ids slot past the step's rows: no token id is 2^32 - 1
EMPTY_ID = 0xFFFFFFFF


def row_words(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """int8 code rows ``(n, K)`` and their f32 scales ``(n,)`` as their bit rows' u16 words ``(n, K / 2 + 2)``: code t's
    bits are leaves 8t .. 8t + 7, then the scale's 32 bits, little-endian (`h1zy.ROW`)."""
    b = np.ascontiguousarray(np.asarray(codes, dtype=np.int8).view(np.uint8))
    b = b.reshape(len(b), -1)
    s = np.ascontiguousarray(np.asarray(scales, "<f4").reshape(len(b))).view(np.uint8).reshape(len(b), 4)
    return np.ascontiguousarray(np.concatenate([b, s], axis=1)).view("<u2").astype(np.uint16)


def chunk_words(codes: np.ndarray, scales: np.ndarray, n: int, slots: int | None = None,
                out: np.ndarray | None = None) -> np.ndarray:
    """int8 code rows ``(r, K)`` and their f32 scales ``(r,)`` as ``n`` leaves a row (`weights.chunks` (K), SPLITK):
    leaf ``n·i + c`` row i's codes ``[c·K/n, (c + 1)·K/n)`` then its scale, `row_words` of the codes' ``(n·r, K/n)``
    view and each scale n times, ``(n·r, K/(2n) + 2)``. At n = 1 it is `row_words`. ``slots``: over that many rows, those
    past r zero codes of scale 0, ``(n·slots, K/(2n) + 2)``. ``out``: written into that u16 array of the words' shape, each
    byte once, else into a new one. ValueError on an n that doesn't divide K into whole u16 words, or fewer slots than
    rows."""
    c = np.asarray(codes, dtype=np.int8)
    r, k = c.shape
    if type(n) is not int or n < 1 or k % (2 * n):
        raise ValueError(f"K = {k} is not {n} chunks of whole u16 words")
    slots = r if slots is None else int(slots)
    if slots < r:
        raise ValueError(f"{r} rows over {slots} slots")
    shape, w = (n * slots, k // (2 * n) + 2), k // n
    if out is None:
        out = np.empty(shape, "<u2")
    elif out.shape != shape or out.dtype != np.dtype("<u2") or not out.flags.c_contiguous:
        raise ValueError(f"out: a contiguous u16 array of {shape}, not {out.dtype} {out.shape}")
    b = out.view(np.uint8)
    b[:n * r, :w] = c.view(np.uint8).reshape(n * r, w)
    b[:n * r, w:] = np.repeat(np.asarray(scales, "<f4").reshape(r), n).view(np.uint8).reshape(n * r, 4)
    b[n * r:] = 0
    return out


def ids_words(ids: Sequence[int | None], slots: int) -> np.ndarray:
    """A step's input token ids over `slots` (M) as u16 words ``(slots, 2)``, each id's u32 little-endian, `EMPTY_ID` for
    None and past them. ValueError on more ids than slots or an id that isn't a u32 below `EMPTY_ID`."""
    if len(ids) > slots:
        raise ValueError(f"{len(ids)} input ids, past the Program's {slots} slots")
    if any(t is not None and not 0 <= int(t) < EMPTY_ID for t in ids):
        raise ValueError(f"an input id is not a u32 below {EMPTY_ID}")
    v = np.full(slots, EMPTY_ID, "<u4")
    v[:len(ids)] = [EMPTY_ID if t is None else int(t) for t in ids]
    return v.view("<u2").reshape(slots, 2).astype(np.uint16)


def receipt_words(receipts: np.ndarray) -> np.ndarray:
    """A call's receipts, one a unit in its tile order, as their bit rows' u16 words: u32 words ``(units, nw)`` (the
    tile's ``receipt``, `h1tile`: per set its lanes' XOR of y, then of z, nw = 2 (KS + m)) each little-endian, low half
    first, so ``(units, 2 nw)``; u16 words ``(units, w)`` as they are."""
    r = np.asarray(receipts)
    if r.ndim != 2 or r.dtype not in (np.uint16, np.uint32):
        raise ValueError(f"a call's receipts are (units, words) u32 or u16 words, not {r.dtype} {r.shape}")
    return np.array(np.ascontiguousarray(r, "<u4" if r.dtype == np.uint32 else "<u2").view("<u2"), np.uint16)


def record_words(z: np.ndarray, y: np.ndarray, slots: int, tile: tuple[int, int]) -> np.ndarray:
    """A call's z/y records over ``slots`` rows as their bit rows' u16 words ``(units, 3 R C)``, one a unit in tile order
    (row block major): its cells' s32 z, row-major, little-endian, then their bf16 y (`h1zy.RECORD`). ``z`` [m, n] s32 and
    ``y`` [m, n] bf16 bits are the served rows'; every later slot is z = 0, y = +0."""
    (r, c), (m, n) = tile, np.shape(z)
    zf, yf = np.zeros((slots, n), "<i4"), np.zeros((slots, n), "<u2")
    zf[:m], yf[:m] = z, np.asarray(y).astype(np.int16).view(np.uint16)

    def cells(v: np.ndarray) -> np.ndarray:
        return v.reshape(slots // r, r, n // c, c).transpose(0, 2, 1, 3).reshape(-1, r * c)
    return np.concatenate([np.ascontiguousarray(cells(zf)).view("<u2"), cells(yf)], axis=1).astype(np.uint16)


def zy_words(records: Any, rows: int, n: int, slots: int, tile: tuple[int, int]) -> Any:
    """`record_words` of a call's records (`pouw_h1_records`' [units, ZY_BYTES] over ``rows`` slots): where the Program's
    tile is the records' unit and they cover all ``slots``, the records' own bytes as u16 words, the same words without
    taking them apart, and of records on a CUDA device a view of them there (`pouw_h1_registered.commit` hashes them in
    place)."""
    if tuple(tile) == ZR.UNIT and rows == slots:
        shape = ((rows // tile[0]) * (n // tile[1]), ZR.ZY_BYTES)
        if getattr(records, "is_cuda", False) and str(records.dtype) == "torch.uint8" and tuple(records.shape) == shape:
            import torch

            return records.contiguous().view(torch.uint16)
        r = ZR.host(records)
        if r.dtype == np.uint8 and r.shape == shape:
            return r.view("<u2")
    return record_words(*ZR.split(ZR.host(records), rows, n), slots, tile)


__all__ = ["EMPTY_ID", "chunk_words", "ids_words", "receipt_words", "record_words", "row_words", "zy_words"]
