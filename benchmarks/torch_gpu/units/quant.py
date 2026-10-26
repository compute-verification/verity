"""The per-row s8 quantizer in CIR-100's form (#4433's ops.quant; the CMP-96 replay's RowAbsMax_v1, QuantS8Scale_v1 and
QuantS8Pair_v1): units_quant, one block per row, and its twin.

OPS, a row x of N bf16 words, RELU on or off (on only for down_quant at the frozen shape):
  v_i     = RELU ? relu16(x_i) : x_i            relu16: +0 where 0 > x_i strictly; -0 and every NaN are themselves
  level 1 = per chunk of min(128, N) words, the integer max of (v_i & 0x7FFF)                     RowAbsMax_v1{128, RELU}
  level k = per chunk of min(128, width) words of level k-1, their integer max, until one word    RowAbsMax_v1{., false}
  amax    = the root (the frozen widths: 16384 -> 128 -> 1, 65536 -> 512 -> 4 -> 1)
  scale   = RN(f32(amax) / 127) if 1 <= f32(amax) <= 0x7F800000 as signed words, else 1.0 (a zero or NaN amax)
                                                                                                  QuantS8Scale_v1
  code_i  = cvt.rni.sat.s8.f32(RN(f32(v_i) / scale))     ties to even, saturated to [-128, 127], NaN 0   QuantS8Pair_v1
The max is over integers (exact, so the tree's shape changes no word; NaN encodings sit above +inf), the division
div.rn.f32.  On the GPU thread j of row r's block takes level 1's chunk j, the next levels' threads their chunks, thread 0
the scale, and every thread writes codes.
"""
from __future__ import annotations

import numpy as np

import words as W

CHUNK = 128


def levels(N: int) -> list[int]:
    """The absmax tree's widths, level 1 first, the root (1) last."""
    out, m = [], N
    while True:
        a = min(CHUNK, m)
        if m < 1 or m % a:
            raise ValueError(f"an absmax tree over {m}: chunks of {a}")
        m //= a
        out.append(m)
        if m == 1:
            return out


def scale(amax: np.ndarray) -> np.ndarray:
    a = W.widen(amax).view(np.int32)
    pos = (a >= 1) & (a <= 0x7F800000)
    return np.where(pos, W.fdiv(a.view(np.uint32), np.uint32(W.F127)), np.uint32(W.ONE)).astype(np.uint32)


def twin(x: np.ndarray, relu: bool) -> dict:
    """x (R, N) bf16 words -> codes (R, N) s8 words (uint8), scale (R,) f32 words and tree (R, T) bf16 words, every
    level's in order, the root last."""
    x = np.asarray(x, np.uint16)
    R, N = x.shape
    v = W.relu16(x, relu)
    lvl, tree = v & np.uint16(0x7FFF), []
    for width in levels(N):
        lvl = lvl.reshape(R, width, -1).max(axis=2).astype(np.uint16)
        tree.append(lvl)
    s = scale(lvl[:, 0])
    codes = W.s8(W.fdiv(W.widen(v), s[:, None]))
    return {"codes": codes, "scale": s, "tree": np.concatenate(tree, axis=1)}


def gpu(x, relu: bool, *, codes=None, scale_out=None, tree=None, units: bool = False):
    """units_quant on x (R, N) bf16 -> (codes (R, N) int8, scale (R,) int32 f32 words); with ``units`` (or ``tree``) also
    tree (R, T) int16 bf16 words.  ``codes``, ``scale_out`` and ``tree`` take the caller's tensors."""
    import torch

    import lib as L
    R, N = x.shape
    lv = levels(N)
    if N % 8 or lv[0] > 1024:
        raise ValueError(f"units_quant takes N a multiple of 8 with at most 1024 level-1 chunks, not N = {N}")
    x = L.words(x, torch.bfloat16, (R, N), "quant x")
    codes = L.out_or_new(codes, torch.int8, (R, N), x, "quant codes")
    sc = L.out_or_new(scale_out, torch.int32, (R,), x, "quant scale")
    L.aligned("quant", x, codes)
    units = units or tree is not None
    tree = L.out_or_new(tree, torch.int16, (R, sum(lv)), x, "quant tree") if units else None
    if R:
        L.lib().launch("units_quant", R, L.threads(lv[0]),
                       [L.ptr(x), L.ptr(codes), L.ptr(sc), L.ptr(tree), L.i32(N), L.i32(sum(lv)), L.i32(int(bool(relu))),
                        L.u(W.F127)])
    return (codes, sc, tree) if units else (codes, sc)
