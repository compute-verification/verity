"""(c) Bit-sliced arithmetic with LOP3 across many words, A = B = 0 (R binary, S' = S + 64 in [0, 128], 8 bits).

- `planes`: bit-sliced over t. A row's segment of R is a 32-bit mask r, and each column's S' a set of 8 bit planes P_jb
  over t (set up once: S repeats). The segment dot is sum_b 2^b popc(r & P_jb) - 64 popc(r): 8 LOP3, 8 POPC, 7 LEA
  (Horner) and one IADD3 that adds the offset and the previous word.
- `csa`: bit-sliced over j. A lane holds 32 columns' bit b of S'_t in one register; for a row (warp-uniform, so a zero
  R_it is a uniform skip, not a mask) the active t's 8-bit numbers are summed by a carry-save tree of full adders (two
  LOP3: XOR3 and majority) and half adders, a ripple adder ends it, a 32 x 32 bit transpose (5 stages of 16 SWAPMOVEs,
  5 instructions each) turns the 32 columns' sums into words, and one IADD3 a word adds the offset and the previous word.
"""
from __future__ import annotations

import numpy as np

from . import dither as D
from . import reference as REF
from .isa import Kernel, M32, U32
from .packed import _offset

TRANSPOSE_OPS = 5 * 16 * 5


def _planes_of(Sp: np.ndarray) -> np.ndarray:
    """(8, n): plane b of column j, bit t = bit b of S'_tj."""
    return np.array([[sum(((int(Sp[t, j]) >> b) & 1) << t for t in range(D.L)) for j in range(Sp.shape[1])]
                     for b in range(8)], dtype=U32)


def planes(inst: D.Instance, fold: str = "li", seed: int = 1) -> REF.Run:
    assert inst.zero
    k = Kernel()
    n = inst.n
    P = _planes_of(inst.U + 64)
    with k.phase("setup"):
        k.add("LOP3", P.size * D.L // 4)
    prev = np.zeros((n, n), dtype=U32)
    W = []
    for l in range(inst.ks):
        r = np.array([sum(int(inst.R[i, l * D.L + t]) << t for t in range(D.L)) for i in range(n)], dtype=U32)
        rr = np.broadcast_to(r[:, None], (n, n))
        v = None
        for b in range(7, -1, -1):
            c = k.popc(k.and2(rr, np.broadcast_to(P[b][None, :], (n, n))))
            v = c if v is None else k.lea(v, c, 1)
        off = _offset(k, inst, l)[:, None]
        prev = k.iadd3(prev, v, np.broadcast_to(off, (n, n)).astype(U32))
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)


def _fa(k: Kernel, a, b, c):
    return k.xor3(a, b, c), k.lop3(a, b, c, lambda x, y, z: (x & y) | (x & z) | (y & z))


def _ha(k: Kernel, a, b):
    return k.xor2(a, b), k.and2(a, b)


def _compress(k: Kernel, cols: dict[int, list]) -> dict[int, list]:
    """Wallace: full adders on every 3 bits of a weight, a half adder on a leftover pair while any weight has > 2."""
    while any(len(v) > 2 for v in cols.values()):
        nxt: dict[int, list] = {}
        for wgt in sorted(cols):
            v = cols[wgt]
            i = 0
            while len(v) - i >= 3:
                s, c = _fa(k, v[i], v[i + 1], v[i + 2])
                nxt.setdefault(wgt, []).append(s)
                nxt.setdefault(wgt + 1, []).append(c)
                i += 3
            if len(v) - i == 2 and len(v) > 2:
                s, c = _ha(k, v[i], v[i + 1])
                nxt.setdefault(wgt, []).append(s)
                nxt.setdefault(wgt + 1, []).append(c)
                i += 2
            nxt.setdefault(wgt, []).extend(v[i:])
        cols = nxt
    return cols


def _ripple(k: Kernel, cols: dict[int, list], width: int) -> list:
    """Two rows to one: a half adder at the lowest weight with two bits, full adders above (XOR3 and majority)."""
    out, carry = [], None
    for wgt in range(width):
        v = list(cols.get(wgt, []))
        if carry is not None:
            v.append(carry)
        if not v:
            out.append(None)
            carry = None
        elif len(v) == 1:
            out.append(v[0])
            carry = None
        elif len(v) == 2:
            s, carry = _ha(k, v[0], v[1])
            out.append(s)
        else:
            s, carry = _fa(k, v[0], v[1], v[2])
            out.append(s)
    return out


def csa(inst: D.Instance, fold: str = "li", seed: int = 1) -> REF.Run:
    assert inst.zero
    k = Kernel()
    n = inst.n
    Sp = (inst.U + 64).astype(np.int64)
    groups = n // 32
    # Q[t][b]: (groups,) words, bit c of word g = bit b of S'_(t, 32 g + c); set up once (S repeats)
    Q = [[np.array([sum(((int(Sp[t, 32 * g + c]) >> b) & 1) << c for c in range(32)) for g in range(groups)],
                   dtype=U32) for b in range(8)] for t in range(D.L)]
    with k.phase("setup"):
        k.add("LOP3", D.L * 8 * groups * 8)
    prev = np.zeros((n, n), dtype=U32)
    W = []
    width = 13
    for l in range(inst.ks):
        seg = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            active = [t for t in range(D.L) if inst.R[i, l * D.L + t]]
            cols = {b: [Q[t][b] for t in active] for b in range(8)}
            bits = _ripple(k, _compress(k, cols), width)
            k.add("LOP3", TRANSPOSE_OPS * groups // 2)
            k.add("SHF", TRANSPOSE_OPS * groups // 2)
            for wgt, v in enumerate(bits):
                if v is None:
                    continue
                for c in range(32):
                    seg[i, c::32][:groups] += ((v >> U32(c)) & U32(1)).astype(np.int64) << wgt
        off = _offset(k, inst, l)[:, None]
        prev = k.iadd3(prev, (seg & M32).astype(U32), np.broadcast_to(off, (n, n)).astype(U32))
        W.append(prev)
    W = np.stack(W)
    return REF.Run(W, REF.receipt(k, W, fold, seed), k)
