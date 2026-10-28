"""Receipts inside attention (`kv.cu`'s `kv_attn` forms 10 and 11, on mma-ws): the sign table, the reference sketches
and the receipts.

~~~text
block      RT tiles of one (sequence, KV head) pair at tile offsets that are multiples of RT; a tile is MT = 32 tokens, and
           a decoder warp's lane l of warp i holds word l % HW of token (32 / HW) i + l // HW of the head's share (K
           words, then V words; HW = 8 at 64-byte words, 4 at 128), WB = 16 S raw (encoded) bytes, S the sketch's steps
vector     warp i's A row r = g + 8e over a block: byte b of the lane's register 4s + 2h + e, lane 4g + tq, tile τ, at
           position p = 32 S τ + 32s + 16h + 4tq + b, so n = 32 S RT (the sketch tag's fragment order); index
           ((pair · blocks + block) · MDW + i) · 16 + r, pair = sequence · KV_HEADS + head
S          16 × n signs: derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": 16, "n": n}), row c the first n/8 bytes of
           stream(("row", c)), bit p (LSB first) 1 → −1 (the sketch tag's `sketch.sign_bits`)
B          unbound (ro-sketch-blake3/v0): S[c][p]. Bound (ro-bound-sketch-blake3/v0, the bound receipt's §1, 64-byte words
           only): byte b of R ⊕ M as int8, register q = 2 blk + h of (warp i, step s, lane l = 4g + tq) for c = 8 blk + g,
           R = z word (512 i + 128 s + 4 l + q) mod ZW of the CTA's tile RLAG back (0 for its first RLAG tiles), M = 0xFF
           where S[c][p] = −1; z is the MMA warps' PV accumulators after that tile (`kv.cu`'s layout)
sketch     v[r][c] = Σ_p e_r[p] · B[p][c], 16 int32
receipt    BLAKE3-keyed(K, u64be(vector) ‖ sketch as 16 int32 LE), K = tags.Blake3Tag(NAME).key(σ)
SHA-512    the half-block receipt in the remote driver's hash (`remote_bandwidth.receipts`' y, seed-first):
           SHA512(Z(SHA_DOMAIN ‖ frame(σ)) ‖ u64be(half) ‖ its 8 sketches as 128 int32 LE), Z zero-padding to one
           128-byte block, so 5 compressions after the prefix's midstate
~~~
"""

from __future__ import annotations

import hashlib

import numpy as np

from verity.core.protocols.bandwidth import tags
from verity.ml.randomness import derive, frame

NAMES = {1: "ro-sketch-blake3/v0", 2: "ro-bound-sketch-blake3/v0"}
SIGN_DOMAIN = "verity/bandwidth/sketch-signs/v0"
SHA_DOMAIN = b"verity/word-codec/kv-receipt/sha512/v0\0"
KS = 16
LANES = 32


def sign_bits(seed: bytes, rc: int, n: int) -> np.ndarray:
    """S's bits, 16 × n uint8 (1 is −1)."""
    key = derive(seed, SIGN_DOMAIN, {"tag": NAMES[rc], "k": KS, "n": n})
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(KS)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def rows(bits: np.ndarray, rt: int, rc: int) -> np.ndarray:
    """`kv_rc_set`'s table, (rt, 128 S) uint32 for 16 × 32 S RT signs: word 128 s + 4 l + q of tile τ is B register
    q = 2 blk + h of step s at lane l = 4g + tq, byte b the sign of column 8 blk + g at position 32 S τ + 32 s + 16 h +
    4 tq + b: 0x01 or 0xFF for +1 or −1 (rc 1), the mask M's 0x00 or 0xFF (rc 2)."""
    steps = bits.shape[1] // (32 * rt)
    if bits.shape != (KS, 32 * steps * rt) or steps < 1 or rc == 2 and steps != 4:
        raise ValueError("a block's signs are 16 × 32 S RT (S = 4 bound)")
    m = bits.reshape(2, 8, rt, steps, 2, 4, 4).transpose(2, 3, 1, 5, 0, 4, 6)  # τ, s, g, tq, blk, h, b
    by = np.where(m == 1, 0xFF, 0x01 if rc == 1 else 0x00).astype(np.uint8)
    n = 128 * steps
    return np.ascontiguousarray(by).reshape(rt, n, 4).view("<u4").reshape(rt, n).astype(np.uint32)


def key_words(seed: bytes, rc: int) -> np.ndarray:
    return np.frombuffer(tags.Blake3Tag(name=NAMES[rc]).key(seed), dtype="<u4").copy()


def vectors(lanes: np.ndarray, rt: int) -> np.ndarray:
    """A CTA's lanes (tiles, MDW, 32, 16 S) uint8, tiles a multiple of rt, as (blocks, MDW, 16, 32 S rt) int8."""
    t, nw, _, wb = lanes.shape
    a = lanes.reshape(t // rt, rt, nw, 8, 4, wb // 16, 2, 2, 4)  # blk, τ, i, g, tq, s, h, e, b
    a = a.transpose(0, 2, 7, 3, 1, 5, 6, 4, 8)  # blk, i, e, g, τ, s, h, tq, b
    return np.ascontiguousarray(a).reshape(t // rt, nw, 16, rt * 2 * wb).view(np.int8)


def multipliers(bits: np.ndarray, rt: int, nw: int, z: np.ndarray | None, lag: int, zw: int) -> np.ndarray:
    """B for each block and warp of a CTA, (blocks, MDW, n, 16) int8; z (tiles, zw) uint32 the z after each tile (None:
    unbound, any S; bound, S = 4)."""
    n = bits.shape[1]
    if z is None:
        s = (1 - 2 * bits.astype(np.int8)).T  # n × 16
        return np.broadcast_to(s, (1, nw, n, KS))
    if n != 128 * rt:
        raise ValueError("the bound form takes 64-byte words")
    t = z.shape[0]
    zt = np.concatenate([np.zeros((lag, zw), np.uint32), z[: t - lag]]) if t > lag else np.zeros((t, zw), np.uint32)
    i, s, lane, q = np.meshgrid(np.arange(nw), np.arange(4), np.arange(LANES), np.arange(4), indexing="ij")
    idx = (512 * i + 128 * s + 4 * lane + q) % zw
    rw = zt[:, idx]  # tiles, i, s, l, q
    rb = np.ascontiguousarray(rw, dtype="<u4").view(np.uint8).reshape(t, nw, 4, 8, 4, 2, 2, 4)  # T, i, s, g, tq, blk, h, b
    m = (0xFF * bits.astype(np.uint8)).reshape(2, 8, rt, 4, 2, 4, 4)  # blk, g, τ, s, h, tq, b
    m = m.transpose(2, 3, 1, 5, 0, 4, 6)  # τ, s, g, tq, blk, h, b
    rb = rb.reshape(t // rt, rt, nw, 4, 8, 4, 2, 2, 4) ^ m[None, :, None]
    bv = rb.view(np.int8).transpose(0, 2, 1, 3, 7, 5, 8, 6, 4)  # blk, i, τ, s, h, tq, b, blk_c, g
    return np.ascontiguousarray(bv).reshape(t // rt, nw, n, KS)


def sketch(lanes: np.ndarray, bits: np.ndarray, rt: int, z: np.ndarray | None = None, lag: int = 2,
           zw: int = 1024) -> np.ndarray:
    """A CTA's sketches, (blocks, MDW, 16, 16) int64: lanes (tiles, MDW, 32, 16 S) uint8, zero-padded to whole
    blocks."""
    v = vectors(lanes, rt).astype(np.float64)
    b = multipliers(bits, rt, lanes.shape[1], z, lag, zw).astype(np.float64)
    return np.rint(v @ b).astype(np.int64)


def receipt(seed: bytes, rc: int, vector: int, sk: np.ndarray) -> bytes:
    return tags.Blake3Tag(name=NAMES[rc]).keyed(seed)(vector, sk.astype("<i4").tobytes())


def sha_prefix(seed: bytes) -> bytes:
    """`kv_rc_sha_set`'s block: SHA_DOMAIN ‖ frame(σ), zero-padded to 128 bytes."""
    p = SHA_DOMAIN + frame(seed)
    if len(p) > 128:
        raise ValueError("the prefix is one SHA-512 block")
    return p + bytes(128 - len(p))


def sha_receipt(seed: bytes, half: int, sk: np.ndarray) -> bytes:
    """A half-block's SHA-512 receipt over its 8 sketches (8, 16)."""
    if sk.shape != (8, 16):
        raise ValueError("a half-block is 8 sketches")
    return hashlib.sha512(sha_prefix(seed) + half.to_bytes(8, "big") + sk.astype("<i4").tobytes()).digest()
