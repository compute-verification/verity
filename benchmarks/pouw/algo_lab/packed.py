"""(b) Packed integer multiplies on the lanes: several products an IMAD in spaced bit fields, and what finishing each exact
word costs. A = B = 0 only (narrow operands); with a general A, `idp4a` is the lane route (4 s8 MACs an instruction).

With S' = S + 64 in [0, 128] (8 bits), a segment's dot is sum_t R_it S'_tj - 64 popc(R_i,seg), each partial at most
32 x 128 = 4096, so a field needs 13 bits.

- `fields`: two columns' S' in one register at bits 0 and 16; IMAD(R_it, s2_t, acc) adds both products a step (R in
  {0, 1}); at the segment's end, PRMT takes each 16-bit field out and one IADD3 adds it, the offset and the previous word.
- `poly`: both sides packed along t at spacing w = 12, G = 3 terms an operand: a = sum R_(i, t+e) 2^(12 e) and
  b = sum S'_(t+e, j) 2^(12 (2 - e)), so bits 24..35 of IMAD.WIDE.U32's 64-bit a b hold the 3-term dot, and the 64-bit
  accumulator carries every group of a half segment (16 steps, at most 2048 a field, so no carry crosses a field) before
  SHF.R.U64 and a LOP3 take the field out.
- `idp4a`: IDP4A.S8.S8 on four t's a step, 8 a word, any A and B.
"""
from __future__ import annotations

import numpy as np

from . import dither as D
from . import reference as REF
from .isa import Kernel, M32, U32, U64


def _offset(k: Kernel, inst: D.Instance, l: int) -> np.ndarray:
    """-64 popc(R_i, seg l) a row: a POPC and an IMAD a row and segment (n of them against n^2 words)."""
    r = inst.R[:, l * D.L:(l + 1) * D.L]
    k.add("POPC", inst.n)
    k.add("IMAD", inst.n)
    return (-64 * r.sum(axis=1)) & M32


def fields(inst: D.Instance, fold: str = "li", seed: int = 1) -> REF.Run:
    assert inst.zero
    k = Kernel()
    n = inst.n
    Sp = inst.U + 64
    s2 = (Sp[:, 0::2] | (Sp[:, 1::2] << 16)).astype(U32)          # (L, n / 2): set up once, S repeats
    with k.phase("setup"):
        k.add("LEA", s2.size)
    prev = np.zeros((n, n), dtype=U32)
    W = []
    for l in range(inst.ks):
        acc = np.zeros((n, n // 2), dtype=U32)
        for t in range(D.L):
            r = inst.R[:, l * D.L + t].astype(U32)[:, None]
            acc = k.imad(np.broadcast_to(r, acc.shape), np.broadcast_to(s2[t], acc.shape), acc)
        off = _offset(k, inst, l)[:, None]
        lo = (acc & U32(0xFFFF)).astype(U32)
        hi = (acc >> U32(16)).astype(U32)
        k.add("PRMT", 2 * acc.size)
        seg = np.empty((n, n), dtype=U32)
        seg[:, 0::2], seg[:, 1::2] = lo, hi
        prev = k.iadd3(prev, seg, np.broadcast_to(off, seg.shape).astype(U32))
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)


def poly(inst: D.Instance, fold: str = "li", seed: int = 1, w: int = 12, G: int = 3, flush: int = 16) -> REF.Run:
    assert inst.zero and w * (G - 1) + 8 <= 32 and flush * 128 < (1 << w)
    k = Kernel()
    n = inst.n
    Sp = (inst.U + 64).astype(np.int64)
    groups = [list(range(s, min(s + G, h + flush))) for h in range(0, D.L, flush) for s in range(h, h + flush, G)]
    bpack = [sum(Sp[t] << (w * (G - 1 - e)) for e, t in enumerate(g)).astype(U32) for g in groups]   # (n,) each
    with k.phase("setup"):
        k.add("LEA", len(groups) * n * (G - 1))
    prev = np.zeros((n, n), dtype=U32)
    W = []
    for l in range(inst.ks):
        seg = np.zeros((n, n), dtype=np.int64)
        for h in range(0, D.L, flush):
            acc = np.zeros((n, n), dtype=U64)
            for g, b in zip(groups, bpack):
                if not h <= g[0] < h + flush:
                    continue
                a = sum(inst.R[:, l * D.L + t].astype(np.int64) << (w * e) for e, t in enumerate(g)).astype(U32)
                with k.phase("setup"):
                    k.add("LEA", n * (len(g) - 1))           # a row's pack: n of them a group, against n^2 IMAD.WIDE
                acc = k.imad_wide(np.broadcast_to(a[:, None], (n, n)), np.broadcast_to(b[None, :], (n, n)), acc)
            f = k.shf_r_wide(acc, w * (G - 1))
            f = k.and2(f, (1 << w) - 1)
            seg += f.astype(np.int64)
        off = _offset(k, inst, l)[:, None]
        k.add("IADD3", n * n)               # prev + f1 + f2 + offset is two IADD3 a word: this one and the next
        prev = k.iadd3(prev, (seg & M32).astype(U32), np.broadcast_to(off, (n, n)).astype(U32))
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)


def idp4a(inst: D.Instance, fold: str = "li", seed: int = 1) -> REF.Run:
    """Any A and B: the int8 operands four to a register (setup), 8 IDP4A a word a segment."""
    k = Kernel()
    REF.setup_operands(k, inst)
    n = inst.n
    At, Bt = inst.At & 0xFF, inst.Bt & 0xFF
    prev = np.zeros((n, n), dtype=U32)
    W = []
    for l in range(inst.ks):
        acc = prev
        for q in range(D.L // 4):
            ts = range(l * D.L + 4 * q, l * D.L + 4 * q + 4)
            a = sum(At[:, t].astype(np.int64) << (8 * e) for e, t in enumerate(ts)).astype(U32)
            b = sum(Bt[t, :].astype(np.int64) << (8 * e) for e, t in enumerate(ts)).astype(U32)
            acc = k.idp4a(np.broadcast_to(a[:, None], (n, n)), np.broadcast_to(b[None, :], (n, n)), acc)
        prev = acc
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)
