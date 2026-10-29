"""pfill's tile (pfill.cuh's `dfill_k<NC>`, dcap.cu's filler unless built with -DDFILL_V1) in Python: the receipt a tile writes, from
the round's coins (c0, c1) and the tile's id, for each tile kind: the s8 tile on ±1 (K = 32), the e2m1 tile on ±1 (K = 64,
-DPFILL_E2M1), the s8 tile on {0,1} bits (K = 32, -DPFILL_BITS, Theorem K's setting) and the bU tile (K = 32, A on {0, 1}
and B on {±1, ..., ±64}, -DPFILL_BU). Written from the matrices with hashlib's SHA-512, as pfill.cuh's host `dfill_ref` is
with its own; `dcap dgate` holds the GPU to that `dfill_ref`, and tests/test_pfill_ref.py holds this file to that
reference built with g++ (pfill_host.cpp).

The cheats against the folds are here too, each as the words it hands the same fold:
`q_receipt` computes Q = a0 d0 + d1 a pair with one product a pair (ram's step reads nothing else), and
`packed_receipt` gets both words of a pair from one packed product (4 A)(33 B0 + B1) and a table (±1 and {0,1} only)."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

for _name, _path in (("dfill_ref", Path(__file__).with_name("dfill_ref.py")),
                     ("receipt_folds", Path(__file__).resolve().parents[1] / "receipts" / "receipt_folds.py")):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, _path)
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
smix, dseed = sys.modules["dfill_ref"].smix, sys.modules["dfill_ref"].dseed
RF = sys.modules["receipt_folds"]  # the receipts reference: the folds' steps, key rules and pfill's lane starts

W, TN = 8, 1024
DOM = 0x5046494C4C2D7631  # "PFILL-v1": the s8 tile
DOM4 = 0x5046494C4C2D6634  # "PFILL-f4": the e2m1 tile
DOM01 = 0x5046494C4C2D6231  # "PFILL-b1": the s8 tile on {0,1} bits
DOM_BU = 0x5046494C4C2D6255  # "PFILL-bU": the bU tile
CTR_B, CTR_KEY = 1 << 56, 2 << 56
M32 = 0xFFFFFFFF
RAM, DUO, XOR = 4, 2, 1
LI, XIN13 = 10, 13  # the r9 folds: NC 4's in a -DPFILL_LI or -DPFILL_XIN13 build
R9 = (LI, XIN13)
S8, E2M1, BITS, BU = 0, 1, 2, 3  # pfill.cuh's PF_S8, PF_E2M1, PF_BITS, PF_BU
KINDS = (S8, E2M1, BITS, BU)
DOMS = {S8: DOM, E2M1: DOM4, BITS: DOM01, BU: DOM_BU}


def dom_of(kind: int) -> int:
    return DOMS[int(kind)]


def digest(c0: int, c1: int, tile: int, ctr: int, dom: int = DOM) -> bytes:
    return hashlib.sha512(struct.pack(">5Q", c0, c1, tile, dom, ctr)).digest()


def block(c0: int, c1: int, tile: int, ctr: int, dom: int = DOM) -> np.ndarray:
    """A block's 16 row words: SHA-512 of the 40-byte (c0, c1, tile, dom, ctr), bytes 4r..4r+3 little-endian, so bit k
    of row r is entry 32 r + k's bit (entry value 1 - 2 bit on ±1, the bit itself on BITS and bU's A)."""
    return np.frombuffer(digest(c0, c1, tile, ctr, dom), "<u4").astype(np.uint32)


def rows(c0: int, c1: int, tile: int, base: int, i: int, kind: int) -> np.ndarray:
    """Fragment or B block i's 16 rows over k < K: one block (s8, bits and bU's A, ctr base + i) or two (e2m1: ctr
    base + 2 i + h for k in 32 h .. 32 h + 31, in the high word for h = 1)."""
    if int(kind) != E2M1:
        return block(c0, c1, tile, base + i, dom_of(kind)).astype(np.uint64)
    lo, hi = (block(c0, c1, tile, base + 2 * i + h, DOM4).astype(np.uint64) for h in (0, 1))
    return lo | (hi << np.uint64(32))


def u_of(t: np.ndarray) -> np.ndarray:
    """bU's B entry from its digest byte t: (t & 0x7F) - 64 + ((t >> 6) & 1), uniform on {±1, ..., ±64}."""
    t = np.asarray(t, np.int64)
    return (t & 0x7F) - 64 + ((t >> 6) & 1)


def bu_columns(c0: int, c1: int, tile: int, tn: int = TN) -> np.ndarray:
    """bU's B, 32 x tn: block b (ctr 2^56 + b) holds columns 2b and 2b + 1, entry (k, 2b + i) from digest byte 32 i + k."""
    t = np.stack([np.frombuffer(digest(c0, c1, tile, CTR_B + b, DOM_BU), np.uint8).reshape(2, 32)
                  for b in range(tn // 2)]).reshape(tn, 32)
    return u_of(t).T


key_ok, key_ok_r9, rotl = RF.key_ok, RF.key_ok_r9, RF.rotl


def keys(rows: np.ndarray, r9: bool = False) -> tuple[int, int, int, int]:
    """(a0, a1, k0, k1): the first two row words that pass key_ok (key_ok_r9 in an LI or XIN13 build), then rows 14 and
    15 (receipt_folds.pfill_keys)."""
    return RF.pfill_keys(rows, r9)


def absorb(nc: int, y, z, d0, d1, d2, d3, a0, a1, k0, k1):
    """One MMA's step of a lane's chain, (y, z) from its words (d0, d1, d2, d3) = rows (g, g + 8) x columns (2t, 2t + 1),
    by the receipts reference's steps; every argument uint32 (arrays or scalars)."""
    if nc == RAM:
        return RF.ram_step(y, d0, d1, d2, d3, a0, a1), z
    if nc == LI:
        return RF.li_step(y, d0, d1, d2, d3, a0, a1, k0, k1), z
    if nc == XIN13:
        return RF.xin13_step(y, d0, d1, d2, d3, a0, a1), z
    if nc == DUO:
        return RF.duo_step(y, z, d0, d1, d2, d3, a0, a1, k0, k1)
    return y ^ d0 ^ d1 ^ d2 ^ d3, z


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    x = x.astype(np.uint64)
    return sum(((x >> np.uint64(i)) & np.uint64(1)) for i in range(64)).astype(np.int64)


def words(a: np.ndarray, b: np.ndarray, kind: int) -> np.ndarray:
    """The committed words of A row bits a and B column bits b: s32 32 - 2 popcount(a ^ b) (s8), the bits of the f32
    64 - 2 popcount(a ^ b) (e2m1), or s32 popcount(a & b) (bits)."""
    if kind == BITS:
        return popcount(a & b).astype(np.uint32)
    if kind == E2M1:
        return (64 - 2 * popcount(a ^ b)).astype(np.float32).view(np.uint32)
    return ((32 - 2 * popcount(a ^ b)) & M32).astype(np.uint32)


def bu_products(ar: np.ndarray, u: np.ndarray) -> np.ndarray:
    """bU's committed words, every warp's rows against every column: C = A u over k < 32 with A[w, r, k] = bit k of ar[w,
    r], as s32 bits ([w, 16 S, tn] uint32; |C| <= 2048)."""
    a = ((ar[..., None] >> np.arange(32, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)
    return ((a @ u) & M32).astype(np.uint32)


def a_rows(c0: int, c1: int, tile: int, steps: int, kind: int, w: int = W) -> np.ndarray:
    """A's row bits: [v, 16 f + i] is row i of warp v's fragment f (bit k: entry k; e2m1: k < 64)."""
    return np.stack([np.concatenate([rows(c0, c1, tile, 0, v * steps + f, kind) for f in range(steps)]
                                    or [np.zeros(0, np.uint64)]) for v in range(w)])


def b_cols(c0: int, c1: int, tile: int, kind: int, tn: int) -> np.ndarray:
    """B's column bits for the bit-coded kinds: bit k of [n] is entry (k, n)."""
    return np.concatenate([rows(c0, c1, tile, CTR_B, b, kind) for b in range(tn // 16)])


def tile_words(c0: int, c1: int, tile: int, steps: int, kind: int = S8, w: int = W) -> np.ndarray:
    """Every committed word of the tile, [v, 16 S, 128 w] uint32: warp v's rows against every column."""
    kind, tn = int(kind), 128 * w
    ar = a_rows(c0, c1, tile, steps, kind, w)
    if kind == BU:
        return bu_products(ar, bu_columns(c0, c1, tile, tn))
    return words(ar[:, :, None], b_cols(c0, c1, tile, kind, tn)[None, None, :], kind)


def receipt(c0: int, c1: int, tile: int, nc: int, steps: int, kind: int = S8, w: int = W,
            r9: bool | None = None) -> int:
    """The tile's receipt: warp v's fragments v S .. v S + S - 1 (16 rows each) against all 128 w columns (w warps: the
    pump's tile is 8, dfill_p's PF_PW), fragment f's outputs absorbed by chain f % 4 of each lane in order of f, then of
    the B fragment j. r9: the keys of an LI or XIN13 build (by default, iff fold `nc` is one of those)."""
    return fold(c0, c1, tile, nc, steps, tile_words(c0, c1, tile, steps, kind, w), kind, w, r9)


def fold(c0: int, c1: int, tile: int, nc: int, steps: int, cw: np.ndarray, kind: int = S8, w: int = W,
         r9: bool | None = None) -> int:
    """The receipt that fold `nc` makes of the words `cw` ([v, 16 S, 128 w], as `tile_words`), with the tile's keys."""
    kind = int(kind)
    tn = 128 * w
    kb = block(c0, c1, tile, CTR_KEY, dom_of(kind))
    a0, a1, k0, k1 = (np.uint32(v) for v in keys(kb, nc in R9 if r9 is None else r9))
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    pos = (np.arange(w)[:, None, None] * 32 + lane[None, :, None]) * 4 + np.arange(4)[None, None, :]
    y, z = (v.astype(np.uint32) for v in RF.pfill_start(pos.astype(np.uint64)))
    jj = np.arange(tn // 8)
    n0 = 8 * jj[:, None] + 2 * t[None, :]  # [j, lane]: the lane's columns 8 j + 2 t and 8 j + 2 t + 1
    ww = np.arange(w)[None, :, None, None]
    for fi in range((steps + 3) // 4):
        f = 4 * fi + np.arange(4)
        live = f < steps
        fc = np.minimum(f, max(steps - 1, 0))
        i0 = 16 * fc[None, :] + g[:, None]  # [lane, c]: the lane's rows 16 f + g and 16 f + g + 8
        d0, d1, d2, d3 = (cw[ww, (i0 + h)[None, None], (n0 + n)[:, None, :, None]] for h in (0, 8) for n in (0, 1))
        for j in range(tn // 8):
            ny, nz = absorb(nc, y, z, d0[j], d1[j], d2[j], d3[j], a0, a1, k0, k1)
            y = np.where(live[None, None, :], ny, y)
            z = np.where(live[None, None, :], nz, z)
    key = dseed(c0, c1, tile)
    h = 0
    v = (z.astype(np.uint64) << np.uint64(32)) | y if nc == DUO else y.astype(np.uint64)
    for p, x in zip(pos.ravel().tolist(), v.ravel().tolist()):
        h ^= smix(smix(key ^ p) ^ x)
    return h


def ho_ok(c0: int, c1: int, tile: int, steps: int, kind: int = S8, w: int = W) -> bool:
    """(HO), LI's layout premise: no hand-off of a lane's chain, (d1, d2) in a
    step or (d3, the next step's d0), joins two words whose rows or columns are equal or complementary (either lets one
    product make both words, the complement's through the other operand's sum). Rows: g and g + 8 of a fragment, and
    g + 8 of fragment f with g of f + 4 (chain f % 4's next); columns: 8 j + 2 t with 8 j + 2 t + 1, and that with
    8 (j + 1) + 2 t (column 2 t after the last j). Bit-coded rows and columns compare as bit vectors, bU's columns (on
    {±1, ..., ±64}) as integer vectors, the complement of u being -u."""
    kind, tn = int(kind), 128 * w
    a = a_rows(c0, c1, tile, steps, kind, w).reshape(w, steps, 16)
    full = np.uint64((1 << (64 if kind == E2M1 else 32)) - 1)
    same = lambda x, y: ((x == y) | (x == (y ^ full))).any()  # noqa: E731
    if same(a[:, :, :8], a[:, :, 8:]) or same(a[:, :-4, 8:], a[:, 4:, :8]):
        return False
    n = np.arange(tn // 8)[:, None] * 8 + 2 * np.arange(4)[None, :]  # [j, t]: column 8 j + 2 t
    pairs = ((n, n + 1), (n + 1, np.roll(n, -1, axis=0)))
    if kind == BU:
        u = bu_columns(c0, c1, tile, tn).T
        return not any(((u[x] == u[y]).all(-1) | (u[x] == -u[y]).all(-1)).any() for x, y in pairs)
    bb = b_cols(c0, c1, tile, kind, tn)
    return not any(same(bb[x], bb[y]) for x, y in pairs)


def int_operands(c0: int, c1: int, tile: int, steps: int, kind: int = S8, w: int = W) -> tuple[np.ndarray, np.ndarray]:
    """A ([v, 16 S, 32]) and B ([32, 128 w]) as the integers the IMMA multiplies, for the integer kinds (s8, bits, bU)."""
    kind, tn = int(kind), 128 * w
    if kind == E2M1:
        raise ValueError("the e2m1 tile's words are f32 bits, not integer sums")
    kk = np.arange(32, dtype=np.uint64)
    a = ((a_rows(c0, c1, tile, steps, kind, w)[..., None] >> kk) & np.uint64(1)).astype(np.int64)
    if kind == BU:
        return a, bu_columns(c0, c1, tile, tn).astype(np.int64)
    b = ((b_cols(c0, c1, tile, kind, tn)[None, :] >> kk[:, None]) & np.uint64(1)).astype(np.int64)
    return (1 - 2 * a, 1 - 2 * b) if kind == S8 else (a, b)


def q_words(c0: int, c1: int, tile: int, steps: int, kind: int = S8, w: int = W) -> np.ndarray:
    """The Q-only cheat's words (`tile_words`' shape): a pair's (columns 2m, 2m + 1) are 0 and Q = sum_k A[r, k] (a B[k,
    2m] + B[k, 2m + 1]) mod 2^32, one product a pair, with a = a0 on a fragment's rows 0..7 and a1 on 8..15. Ram's step
    (rotl(y, 7) + d0) a0 + d1 = rotl(y, 7) a0 + Q reads nothing else, so `fold(RAM)` of these is the tile's receipt."""
    a, b = int_operands(c0, c1, tile, steps, kind, w)
    a0, a1 = keys(block(c0, c1, tile, CTR_KEY, dom_of(kind)))[:2]
    cw = np.zeros(a.shape[:2] + (b.shape[1],), np.uint32)
    lo = np.arange(a.shape[1]) % 16 < 8
    for key, r in ((a0, lo), (a1, ~lo)):
        bq = (key * b[:, 0::2] + b[:, 1::2]) & M32
        cw[:, r, 1::2] = (a[:, r] @ bq) & M32
    return cw


def q_receipt(c0: int, c1: int, tile: int, nc: int, steps: int, kind: int = S8, w: int = W) -> int:
    """What the Q-only cheat submits under fold `nc`: the fold of `q_words`."""
    return fold(c0, c1, tile, nc, steps, q_words(c0, c1, tile, steps, kind, w), kind, w)


def packed_words(c0: int, c1: int, tile: int, steps: int, kind: int = S8, w: int = W) -> np.ndarray:
    """The packed cheat's words: one IMMA word a pair, P4 = sum_k (4 A[r, k]) (33 B[k, 2m] + B[k, 2m + 1]) with both
    operands in s8, and a table from P4 to the pair (d0, d1), so every word from half the products. Only the ±1 and {0,1}
    tiles pack: bU's 33 U0 + U1 reaches 2176, and c U0 + U1 leaves s8 for every c >= 1."""
    kind = int(kind)
    if kind not in (S8, BITS):
        raise ValueError(f"kind {kind} doesn't pack: its B entries leave s8 at 33 B0 + B1")
    a, b = int_operands(c0, c1, tile, steps, kind, w)
    a4, bp = 4 * a, 33 * b[:, 0::2] + b[:, 1::2]
    assert max(abs(a4).max(), abs(bp).max()) <= 127
    p4 = a4 @ bp
    vals = np.arange(-32, 33, 2) if kind == S8 else np.arange(33)
    x, y = (v.ravel() for v in np.meshgrid(vals, vals, indexing="ij"))
    key = 4 * (33 * x + y)
    lo = int(key.min())
    lut = np.full((2, int(key.max()) - lo + 1), 1 << 40, np.int64)
    assert len(set(key.tolist())) == key.size
    lut[0, key - lo], lut[1, key - lo] = x, y
    d = lut[:, p4 - lo]
    assert (d != 1 << 40).all()
    cw = np.zeros(a.shape[:2] + (b.shape[1],), np.uint32)
    cw[..., 0::2], cw[..., 1::2] = d[0] & M32, d[1] & M32
    return cw


def packed_receipt(c0: int, c1: int, tile: int, nc: int, steps: int, kind: int = S8, w: int = W) -> int:
    """What the packed cheat submits under fold `nc`: the fold of `packed_words`."""
    return fold(c0, c1, tile, nc, steps, packed_words(c0, c1, tile, steps, kind, w), kind, w)
